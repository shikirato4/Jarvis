from __future__ import annotations

import pytest

from jarvis.config import Settings
from jarvis.conversation_context import (
    ConservativeTokenEstimator,
    ContextFragment,
    ContextBudgetError,
    ContextSource,
    ContextOverflowError,
    ConversationContextBuilder,
    ConversationContextRequest,
    context_fragments_from_retrieved_context,
)
from jarvis.memory_semantic.base import RetrievedChunk, RetrievedContext
from jarvis.models_runtime.base import ModelMedia, ModelMessage, ModelRequest, ModelToolCall


def test_conservative_estimator_counts_text_and_tool_payload() -> None:
    estimator = ConservativeTokenEstimator()
    plain = ModelMessage(role="user", content="hola mundo")
    tool = ModelMessage(
        role="assistant",
        tool_calls=[
            ModelToolCall(
                id="c1",
                name="calculator.evaluate",
                arguments={"expression": "27*83"},
            )
        ],
    )

    assert estimator.estimate_message(plain) > 0
    assert estimator.estimate_message(tool) > estimator.estimate_message(
        ModelMessage(role="assistant")
    )


def test_context_budget_rejects_output_reservation_that_consumes_window() -> None:
    with pytest.raises(ContextBudgetError, match="reserve"):
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="policy"),
            current_message=ModelMessage(role="user", content="hola"),
            max_context_tokens=128,
            reserve_output_tokens=128,
            safety_margin_tokens=16,
        )


def test_builder_preserves_source_ordering_and_deduplicates_memory() -> None:
    builder = ConversationContextBuilder()
    tool_call = ModelMessage(
        role="assistant",
        tool_calls=[
            ModelToolCall(
                id="call-1",
                name="calculator.evaluate",
                arguments={"expression": "781 * 94"},
            )
        ],
    )
    tool_result = ModelMessage(
        role="tool",
        content='{"value":73414}',
        name="calculator.evaluate",
        tool_call_id="call-1",
    )

    built = builder.build(
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="Jarvis policy"),
            history=[
                ModelMessage(role="user", content="Haz un calculo"),
                ModelMessage(role="assistant", content="Voy a calcularlo."),
            ],
            tool_observations=[tool_call, tool_result],
            attachment_context=[
                ContextFragment(
                    fragment_id="attachment-1",
                    source=ContextSource.ATTACHMENT,
                    content="Tabla adjunta: valor 17",
                )
            ],
            memory_candidates=[
                ContextFragment(
                    fragment_id="memory-low",
                    source=ContextSource.MEMORY,
                    content="El usuario prefiere CSV.",
                    score=0.2,
                ),
                ContextFragment(
                    fragment_id="memory-high",
                    source=ContextSource.MEMORY,
                    content="  El usuario prefiere CSV.  ",
                    score=0.9,
                ),
            ],
            current_message=ModelMessage(role="user", content="Continua"),
            max_context_tokens=4096,
            reserve_output_tokens=512,
            safety_margin_tokens=128,
        )
    )

    assert [message.role for message in built.messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "system",
        "system",
        "user",
    ]
    assert built.stats.history_message_count == 2
    assert built.stats.tool_observation_count == 1
    assert built.stats.attachment_context_count == 1
    assert built.stats.memory_candidate_count == 2
    assert built.stats.memory_selected_count == 1
    assert "memory-high" in built.messages[-2].metadata["fragment_ids"]


def test_builder_telemetry_contains_counts_but_no_private_content() -> None:
    private_sentinel = "PRIVATE-CONTEXT-91"
    events: list[tuple[str, dict[str, object]]] = []
    builder = ConversationContextBuilder(
        event_sink=lambda name, payload: events.append((name, payload))
    )

    built = builder.build(
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="policy"),
            history=[ModelMessage(role="user", content=private_sentinel)],
            attachment_context=[
                ContextFragment(
                    fragment_id="private-path-must-not-leak",
                    source=ContextSource.ATTACHMENT,
                    content=f"attachment {private_sentinel}",
                    metadata={"path": "C:/private/file.txt"},
                )
            ],
            current_message=ModelMessage(role="user", content="recall"),
            max_context_tokens=2048,
            reserve_output_tokens=256,
            correlation_id="context-test-1",
        )
    )

    assert built.stats.history_message_count == 1
    assert [name for name, _payload in events] == ["conversation_context_built"]
    event_payload = events[0][1]
    assert event_payload["correlation_id"] == "context-test-1"
    assert event_payload["history_message_count"] == 1
    assert event_payload["attachment_context_count"] == 1
    assert "build_latency_ms" in event_payload
    serialized_payload = repr(events)
    assert private_sentinel not in serialized_payload
    assert "private-path-must-not-leak" not in serialized_payload
    assert "C:/private/file.txt" not in serialized_payload


def test_twenty_turns_survive_tools_attachments_and_compaction() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    builder = ConversationContextBuilder(
        event_sink=lambda name, payload: events.append((name, payload))
    )
    opening = ModelMessage(
        role="user",
        content="Durante esta conversacion recuerda que la palabra es COBALTO-721.",
        metadata={"context_critical": True},
    )
    twenty_turns = [
        message
        for index in range(20)
        for message in (
            ModelMessage(
                role="user",
                content=f"Pregunta de relleno numero {index}: " + "detalle " * 8,
            ),
            ModelMessage(
                role="assistant",
                content=f"Respuesta de relleno numero {index}: " + "respuesta " * 8,
            ),
        )
    ]
    tool_call = ModelMessage(
        role="assistant",
        tool_calls=[
            ModelToolCall(
                id="calc-compaction-1",
                name="calculator.evaluate",
                arguments={"expression": "27 * 83"},
            )
        ],
    )
    tool_result = ModelMessage(
        role="tool",
        content='{"value":2241}',
        name="calculator.evaluate",
        tool_call_id="calc-compaction-1",
    )

    built = builder.build(
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="Jarvis policy"),
            history=[opening, *twenty_turns],
            tool_observations=[tool_call, tool_result],
            attachment_context=[
                ContextFragment(
                    fragment_id="attachment-compaction-1",
                    source=ContextSource.ATTACHMENT,
                    content="El archivo adjunto contiene la categoria azul.",
                )
            ],
            memory_candidates=[
                ContextFragment(
                    fragment_id="memory-compaction-1",
                    source=ContextSource.MEMORY,
                    content="El usuario prefiere respuestas breves.",
                    score=0.8,
                )
            ],
            current_message=ModelMessage(role="user", content="¿Cual era la palabra?"),
            max_context_tokens=900,
            reserve_output_tokens=160,
            safety_margin_tokens=64,
            correlation_id="compaction-1",
        )
    )

    assert any("COBALTO-721" in message.content for message in built.messages)
    assert any(
        call.id == "calc-compaction-1"
        for message in built.messages
        for call in message.tool_calls
    )
    assert any(message.tool_call_id == "calc-compaction-1" for message in built.messages)
    assert any("categoria azul" in message.content for message in built.messages)
    assert built.stats.context_compacted is True
    assert built.stats.context_truncated is False
    assert built.stats.estimated_context_tokens <= 676
    assert [name for name, _payload in events] == [
        "conversation_context_built",
        "context_compacted",
    ]


def test_protected_context_overflow_is_explicit_and_private() -> None:
    sentinel = "NEVER-LEAK-OVERFLOW-77"
    builder = ConversationContextBuilder()

    with pytest.raises(ContextOverflowError) as raised:
        builder.build(
            ConversationContextRequest(
                system_message=ModelMessage(role="system", content="policy"),
                history=[
                    ModelMessage(
                        role="user",
                        content=f"recuerda {sentinel} " + "dato " * 600,
                        metadata={"context_critical": True},
                    )
                ],
                current_message=ModelMessage(role="user", content="recall"),
                max_context_tokens=256,
                reserve_output_tokens=64,
                safety_margin_tokens=32,
            )
        )

    assert sentinel not in str(raised.value)
    assert "estimated_tokens=" in str(raised.value)


def test_omitted_large_attachment_sets_explicit_truncation_event() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    builder = ConversationContextBuilder(
        event_sink=lambda name, payload: events.append((name, payload))
    )

    built = builder.build(
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="policy"),
            attachment_context=[
                ContextFragment(
                    fragment_id="large-attachment",
                    source=ContextSource.ATTACHMENT,
                    content="adjunto " * 800,
                )
            ],
            current_message=ModelMessage(role="user", content="resume el adjunto"),
            max_context_tokens=320,
            reserve_output_tokens=64,
            safety_margin_tokens=32,
        )
    )

    assert built.stats.context_compacted is True
    assert built.stats.context_truncated is True
    assert built.stats.attachment_context_count == 1
    assert [name for name, _payload in events] == [
        "conversation_context_built",
        "context_compacted",
        "context_truncated",
    ]


def test_finalize_model_request_preserves_attachment_media_and_safe_stats() -> None:
    builder = ConversationContextBuilder()
    private_bytes = b"private-image-bytes"
    request = ModelRequest(
        task_type="assistant",
        prompt="Analiza el archivo",
        messages=[
            ModelMessage(role="system", content="policy"),
            ModelMessage(
                role="user",
                content="Analiza el archivo\n\nContenido derivado: categoria azul",
            ),
        ],
        media=[
            ModelMedia(
                file_id="image-1",
                category="image",
                mime_type="image/png",
                safe_name="image.png",
                data=private_bytes,
            )
        ],
        max_tokens=256,
        metadata={"source": "test"},
        correlation_id="finalize-1",
    )

    finalized = builder.finalize_model_request(
        request,
        history=[
            ModelMessage(role="user", content="mensaje anterior"),
            ModelMessage(role="assistant", content="respuesta anterior"),
        ],
        attachment_context=[
            ContextFragment(
                fragment_id="attachment-safe-1",
                source=ContextSource.ATTACHMENT,
                content="categoria azul",
            )
        ],
        max_context_tokens=2048,
        safety_margin_tokens=128,
    )

    assert [message.role for message in finalized.messages[:4]] == [
        "system",
        "user",
        "assistant",
        "system",
    ]
    assert "categoria azul" in finalized.messages[-1].content
    assert finalized.media[0].data == private_bytes
    assert finalized.prompt == "Analiza el archivo"
    context_metadata = finalized.metadata["conversation_context"]
    assert context_metadata["history_message_count"] == 2
    assert context_metadata["attachment_context_count"] == 1
    assert "categoria azul" not in repr(context_metadata)
    serialized = finalized.model_dump(mode="json")
    assert "media" not in serialized
    assert "private-image-bytes" not in repr(finalized)


def test_retrieved_context_adapter_is_ranked_bounded_and_path_private() -> None:
    context = RetrievedContext(
        query="preferencias",
        strategy="hybrid",
        chunks=[
            RetrievedChunk(
                chunk_id="lower",
                document_id="doc-1",
                collection_name="memory",
                source_path="C:/private/lower.txt",
                source_type="note",
                content="preferencia secundaria",
                score=0.4,
                rank=2,
            ),
            RetrievedChunk(
                chunk_id="higher",
                document_id="doc-2",
                collection_name="memory",
                source_path="C:/private/higher.txt",
                source_type="note",
                content="preferencia principal",
                score=0.9,
                rank=1,
            ),
        ],
        total_chunks=2,
    )

    fragments = context_fragments_from_retrieved_context(context, limit=1)

    assert [fragment.fragment_id for fragment in fragments] == ["higher"]
    assert fragments[0].content == "preferencia principal"
    assert fragments[0].score == 0.9
    assert "source_path" not in fragments[0].metadata
    assert "C:/private" not in repr(fragments)


def test_context_budget_settings_have_provider_neutral_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.conversation_context_max_tokens == 16_384
    assert settings.conversation_context_safety_margin_tokens == 512
    assert settings.conversation_context_recent_turns == 8
    assert settings.conversation_context_memory_top_k == 5
    assert settings.conversation_context_attachment_char_budget == 12_000
