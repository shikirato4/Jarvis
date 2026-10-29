from __future__ import annotations

from collections.abc import Iterator

from jarvis.chat_optimization import (
    is_extended_long_form_request,
    select_chat_generation_profile,
)
from jarvis.config import Settings
from jarvis.models_runtime.base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
    StreamChunk,
)
from jarvis.models_runtime.output_completion import (
    OutputCompletionPolicy,
    estimate_request_tokens,
    estimated_tokens_from_text,
    merge_continuation_text,
)


def _settings() -> Settings:
    return Settings(_env_file=None, local_runtime_context_size=8192)


def _request(
    *,
    profile: str = "default",
    max_tokens: int | None = None,
    adaptive: bool = True,
    auto_continue: bool = True,
    context_tokens: int = 8192,
) -> ModelRequest:
    return ModelRequest(
        task_type="assistant",
        messages=[ModelMessage(role="user", content="explica el tema")],
        max_tokens=max_tokens,
        metadata={
            "generation_profile": profile,
            "adaptive_output_budget": adaptive,
            "auto_continue_text": auto_continue,
            "output_context_tokens": context_tokens,
        },
        correlation_id="completion-test",
    )


def _response(
    content: str,
    finish_reason: str,
    *,
    completion_tokens: int | None = None,
    tool_calls: list[ModelToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse(
        provider_name="qwen_local",
        provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant",
        model_name="Qwen3.5-9B-IQ4_XS",
        content=content,
        tool_calls=tool_calls or [],
        latency_ms=10,
        metadata={
            "finish_reason": finish_reason,
            "stream_closed_normally": True,
            "incomplete": finish_reason == "length",
            "completion_contract_enforced": True,
            "completion_tokens": completion_tokens,
        },
    )


def test_adaptive_budget_distinguishes_short_normal_long_and_preserves_exact_cap() -> None:
    policy = OutputCompletionPolicy(_settings())

    short = policy.prepare_request(_request(profile="fast"))
    normal = policy.prepare_request(_request(profile="default"))
    long_form = policy.prepare_request(_request(profile="detailed"))
    explicit = policy.prepare_request(
        _request(max_tokens=260, adaptive=False, auto_continue=False)
    )
    assert short.max_tokens == 160
    assert normal.max_tokens == 768
    assert long_form.max_tokens == 2048
    assert explicit.max_tokens == 260
    assert explicit.metadata["output_budget"]["owner"] == "caller"


def test_generation_profile_uses_explanatory_structure_not_only_prompt_length() -> None:
    settings = _settings()

    short = select_chat_generation_profile("hola", settings)
    normal = select_chat_generation_profile(
        "Explica cómo funciona un agujero negro.", settings
    )
    long_form = select_chat_generation_profile(
        "Explícame detalladamente cómo hacer composta doméstica, incluyendo "
        "materiales, proceso, errores comunes y cómo saber cuándo está lista.",
        settings,
    )

    assert short.name == "fast"
    assert normal.name == "default"
    assert long_form.name == "detailed"


def test_requested_brief_and_extensive_depth_select_different_profiles() -> None:
    settings = _settings()

    brief = select_chat_generation_profile(
        "Explica brevemente qué es la computación cuántica.", settings
    )
    extensive = select_chat_generation_profile(
        "Explícame extensamente cómo funciona la computación cuántica, desde sus "
        "fundamentos hasta sus aplicaciones, desarrollando cada concepto con ejemplos.",
        settings,
    )
    explicit = select_chat_generation_profile(
        "Redacta un informe de aproximadamente 2,000 palabras sobre astronomía.",
        settings,
        web_used=True,
    )

    assert brief.name == "fast"
    assert extensive.name == "extended"
    assert explicit.name == "extended"


def test_extended_long_form_classification_is_selective_and_language_independent() -> None:
    settings = _settings()
    spanish = (
        "explicame muy detaslladamente y muy claramente, muy muy a fondo con "
        "formulas y matematicas la relatividad general y la historia"
    )
    english = (
        "Give me a very deep, mathematically detailed explanation of quantum "
        "mechanics, including its history, core equations, interpretations, "
        "experiments, and worked conceptual examples."
    )
    ordinary_long = (
        "Explícame detalladamente cómo hacer composta doméstica, incluyendo "
        "materiales, proceso, errores comunes y cómo saber cuándo está lista."
    )
    typo_heavy = (
        "bro podrias decirme deetalladamente y epxlicarme como es la "
        "relatividad general de einstein explicarme la historia detallada, y "
        "explicarme cada cosa con formulas, imagina que eres el mejor porfesor "
        "del mundo e imagina que yo soy el estudiante mas tonto del mundo"
    )

    assert is_extended_long_form_request(spanish) is True
    assert is_extended_long_form_request(english) is True
    assert is_extended_long_form_request(typo_heavy) is True
    assert select_chat_generation_profile(spanish, settings).name == "extended"
    assert select_chat_generation_profile(english, settings).name == "extended"
    assert select_chat_generation_profile(ordinary_long, settings).name == "detailed"
    assert select_chat_generation_profile(
        "¿Qué es la relatividad general en dos frases?", settings
    ).name == "fast"


def test_hard_reasoning_gets_detailed_budget_without_becoming_extended() -> None:
    settings = _settings()
    prompt = (
        "Resuelve cuidadosamente este problema y demuestra si para todo entero "
        "n mayor que 1 existe un primo entre n y 2n."
    )

    profile = select_chat_generation_profile(prompt, settings)

    assert profile.name == "detailed"
    request = _request(profile=profile.name).model_copy(
        update={
            "metadata": {
                **_request(profile=profile.name).metadata,
                "reasoning_preference": "xhigh",
                "reasoning_preference_requested": "auto",
                "reasoning_preference_explicit": False,
            }
        }
    )
    prepared = OutputCompletionPolicy(settings).prepare_request(request)
    assert prepared.max_tokens == 2048
    assert prepared.metadata["reasoning_preference"] == "xhigh"


def test_extended_budget_is_larger_but_reserves_context_and_keeps_explicit_cap() -> None:
    policy = OutputCompletionPolicy(_settings())
    extended = policy.prepare_request(_request(profile="extended"))
    explicit = policy.prepare_request(
        _request(
            profile="extended",
            max_tokens=260,
            adaptive=False,
            auto_continue=False,
        )
    )
    explicit_768 = policy.prepare_request(
        _request(
            profile="extended",
            max_tokens=768,
            adaptive=False,
            auto_continue=True,
        )
    )

    assert extended.max_tokens == 4096
    assert extended.metadata["output_budget"]["followup_reserve_tokens"] == 1536
    assert policy.total_response_limit(extended) == 24_000
    assert explicit.max_tokens == 260
    assert explicit.metadata["output_budget"]["owner"] == "caller"
    assert explicit.metadata["disable_auto_continuation"] is True
    assert explicit_768.max_tokens == 768
    assert explicit_768.metadata["disable_auto_continuation"] is True


def test_extended_reasoning_only_length_is_counted_and_continuation_focuses_output() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    continuation_requests: list[ModelRequest] = []

    def infer_next(next_request: ModelRequest) -> ModelResponse:
        continuation_requests.append(next_request)
        return _response("Respuesta visible completa.", "stop", completion_tokens=80)

    result = policy.complete_response(
        request,
        _response("", "length"),
        infer_next=infer_next,
    )

    assert result.metadata["segment_completion_tokens"] == [4096, 80]
    assert result.metadata["segment_max_tokens"] == [4096, 2500]
    assert result.metadata["output_request_class"] == "EXTENDED_LONG_FORM"
    assert result.metadata["finish_reason"] == "stop"
    assert continuation_requests[0].metadata["reasoning_preference"] == "off"
    assert "Finish the current paragraph" in continuation_requests[0].messages[-1].content


def test_auto_hard_reasoning_starvation_recovers_at_low_but_explicit_xhigh_wins() -> None:
    policy = OutputCompletionPolicy(_settings())
    auto = policy.prepare_request(
        _request(profile="detailed").model_copy(
            update={
                "metadata": {
                    **_request(profile="detailed").metadata,
                    "reasoning_preference": "xhigh",
                    "reasoning_preference_requested": "auto",
                    "reasoning_preference_explicit": False,
                }
            }
        )
    )
    explicit = auto.model_copy(
        update={
            "metadata": {
                **auto.metadata,
                "reasoning_preference_requested": "xhigh",
                "reasoning_preference_explicit": True,
            }
        }
    )

    auto_decision = policy.continuation_request(
        auto, "", generated_tokens=2048, attempt=1, total_limit=3686
    )
    explicit_decision = policy.continuation_request(
        explicit, "", generated_tokens=2048, attempt=1, total_limit=3686
    )

    assert auto_decision.request is not None
    assert auto_decision.request.metadata["reasoning_preference"] == "low"
    assert auto_decision.request.metadata["reasoning_starvation_recovery"] is True
    assert explicit_decision.request is not None
    assert explicit_decision.request.metadata["reasoning_preference"] == "xhigh"
    assert "reasoning_starvation_recovery" not in explicit_decision.request.metadata


def test_extended_continuation_reaches_eight_request_bound_truthfully() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))

    result = policy.complete_response(
        request,
        _response("Inicio", "length"),
        infer_next=lambda _request: _response(" continuación", "length"),
    )

    assert result.metadata["auto_continuation_count"] == 8
    assert result.metadata["model_request_count"] == 9
    assert result.metadata["segment_max_tokens"] == [4096] + [2500] * 8
    assert result.metadata["provider_completion_tokens"] == 24_096
    assert result.metadata["total_response_token_limit"] == 24_000
    assert result.metadata["rolling_continuation_count"] == 8
    assert result.metadata["continuation_stop_reason"] == "max_continuations_reached"
    assert result.metadata["finish_reason"] == "length"


def test_extended_auto_reduces_exposition_reasoning_but_explicit_modes_win() -> None:
    policy = OutputCompletionPolicy(_settings())
    auto = _request(profile="extended").model_copy(
        update={
            "metadata": {
                **_request(profile="extended").metadata,
                "reasoning_preference": "xhigh",
                "reasoning_preference_requested": "auto",
                "reasoning_preference_explicit": False,
            }
        }
    )

    prepared_auto = policy.prepare_request(auto)

    assert prepared_auto.metadata["reasoning_preference"] == "off"
    assert prepared_auto.metadata["educational_auto_reasoning_reduced"] is True
    for explicit in ("off", "low", "medium", "xhigh"):
        request = auto.model_copy(
            update={
                "metadata": {
                    **auto.metadata,
                    "reasoning_preference": explicit,
                    "reasoning_preference_requested": explicit,
                    "reasoning_preference_explicit": True,
                }
            }
        )
        assert policy.prepare_request(request).metadata["reasoning_preference"] == explicit

    hard_reasoning = auto.model_copy(
        update={
            "metadata": {
                **auto.metadata,
                "generation_profile": "detailed",
            }
        }
    )
    assert policy.prepare_request(hard_reasoning).metadata["reasoning_preference"] == "xhigh"


def test_rolling_state_is_bounded_public_only_and_preserves_tail_structure() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = _request(profile="extended").model_copy(
        update={
            "messages": [
                ModelMessage(role="system", content="public system"),
                ModelMessage(role="user", content="teach the whole public topic"),
            ],
            "metadata": {
                **_request(profile="extended").metadata,
                "reasoning_content": "PRIVATE_COT_MUST_NOT_PERSIST",
            },
        }
    )
    public = (
        "# Historia\n\nLa historia pública comienza aquí. "
        + "dato visible. " * 1800
        + "\n\n## Matemáticas\n\nDefinimos $g_{\\mu\\nu}$ como la métrica.\n\n"
        + "```python\nprint('tail')\n"
        + "$$\nG_{\\mu\\nu}=8\\pi T_{\\mu\\nu}\n"
    )

    state = policy.build_rolling_state(request, public)
    decision = policy.continuation_request(
        policy.prepare_request(request),
        public,
        generated_tokens=estimated_tokens_from_text(public),
        attempt=1,
        total_limit=24_000,
    )

    assert estimated_tokens_from_text(state.public_summary) <= 768
    assert estimated_tokens_from_text(state.recent_tail) <= 1000
    assert state.recent_tail.endswith("G_{\\mu\\nu}=8\\pi T_{\\mu\\nu}\n")
    assert "PRIVATE_COT_MUST_NOT_PERSIST" not in repr(state)
    assert "open_code_fence=True" in state.structural_state
    assert "open_display_math=True" in state.structural_state
    assert decision.request is not None
    serialized = "\n".join(message.content for message in decision.request.messages)
    assert "PRIVATE_COT_MUST_NOT_PERSIST" not in serialized
    assert public not in serialized
    assert "$g_{\\mu\\nu}$" in serialized or "G_{\\mu\\nu}" in serialized
    assert estimate_request_tokens(decision.request) + int(decision.request.max_tokens or 0) + 1024 <= 8192


def test_extended_soft_instruction_and_visible_hard_ceiling_are_deterministic() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    at_soft = "s" * (16_000 * 4)
    soft = policy.continuation_request(
        request,
        at_soft,
        generated_tokens=16_000,
        attempt=1,
        total_limit=24_000,
    )
    hard = policy.continuation_request(
        request,
        "h" * (23_900 * 4),
        generated_tokens=23_900,
        attempt=8,
        total_limit=24_000,
    )

    assert soft.request is not None
    assert "already extensive" in soft.request.messages[-1].content
    assert hard.request is None
    assert hard.reason == "visible_hard_ceiling"
    assert hard.remaining_total_tokens == 100


def test_extended_visible_accumulation_stops_at_24k_without_a_large_request() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    segment_index = 0

    def infer(_next_request: ModelRequest) -> ModelResponse:
        nonlocal segment_index
        segment_index += 1
        return _response(
            chr(ord("A") + segment_index) * 10_000,
            "length",
            completion_tokens=2500,
        )

    result = policy.complete_response(
        request,
        _response("A" * 16_000, "length", completion_tokens=4096),
        infer_next=infer,
    )

    assert result.metadata["visible_response_tokens_estimated"] == 24_000
    assert result.metadata["logical_response_remaining"] == 0
    assert result.metadata["auto_continuation_count"] == 8
    assert max(result.metadata["segment_max_tokens"]) == 4096


def test_extended_loop_gets_one_correction_then_stops_if_repeated_again() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    repeated = "Este párrafo público ya fue explicado con suficiente detalle. " * 8
    calls: list[ModelRequest] = []

    def infer(next_request: ModelRequest) -> ModelResponse:
        calls.append(next_request)
        return _response(repeated, "length", completion_tokens=200)

    result = policy.complete_response(
        request,
        _response(repeated, "length", completion_tokens=200),
        infer_next=infer,
    )

    assert len(calls) == 2
    assert calls[1].metadata["rolling_loop_correction"] is True
    assert result.content == repeated
    assert result.metadata["loop_detection_count"] == 2
    assert result.metadata["loop_correction_attempted"] is True
    assert result.metadata["continuation_stop_reason"] == "repetition_loop"
    assert result.metadata["finish_reason"] == "length"


def test_extended_recalculates_context_and_preserves_math_markdown() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    continuation_requests: list[ModelRequest] = []
    responses = iter(
        [
            _response("\\mu\\nu} = 8\\pi T_{\\mu\\nu}\n$$\n", "length", completion_tokens=100),
            _response("\n## Conclusión\n\nLa teoría queda resumida.", "stop", completion_tokens=80),
        ]
    )

    def infer_next(next_request: ModelRequest) -> ModelResponse:
        continuation_requests.append(next_request)
        return next(responses)

    result = policy.complete_response(
        request,
        _response("## Matemáticas\n\n$$\nG_{", "length", completion_tokens=100),
        infer_next=infer_next,
    )

    remaining = [
        item.metadata["continuation_remaining_context_tokens"]
        for item in continuation_requests
    ]
    assert remaining[1] < remaining[0]
    assert result.content == (
        "## Matemáticas\n\n$$\nG_{\\mu\\nu} = 8\\pi T_{\\mu\\nu}\n$$\n"
        "\n## Conclusión\n\nLa teoría queda resumida."
    )
    assert result.metadata["finish_reason"] == "stop"


def test_agent_budget_estimates_only_effective_compact_tool_subset() -> None:
    policy = OutputCompletionPolicy(_settings())
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"test.action_{index}",
                "description": "x" * 300,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
        for index in range(40)
    ]
    request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="ejecuta una acción")],
        tools=tools,
        metadata={
            "adaptive_output_budget": True,
            "agent_phase": "structured_tool_call",
        },
    )

    prepared = policy.prepare_request(request)

    assert prepared.max_tokens == 1024
    assert prepared.metadata["output_budget"]["estimated_input_tokens"] < 8192


def test_text_length_auto_continues_and_deduplicates_confident_overlap() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="detailed"))
    calls: list[ModelRequest] = []

    def infer_next(next_request: ModelRequest) -> ModelResponse:
        calls.append(next_request)
        return _response("párrafo repetido y final.", "stop", completion_tokens=80)

    result = policy.complete_response(
        request,
        _response("Inicio. párrafo repetido", "length", completion_tokens=100),
        infer_next=infer_next,
    )

    assert result.content == "Inicio. párrafo repetido y final."
    assert result.metadata["auto_continuation_count"] == 1
    assert result.metadata["model_request_count"] == 2
    assert result.metadata["segment_finish_reasons"] == ["length", "stop"]
    assert result.metadata["deduplicated_overlap_chars"] == len(
        "párrafo repetido"
    )
    assert calls[0].tools == []


def test_normal_stop_uses_one_generation_without_continuation() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request())

    result = policy.complete_response(
        request,
        _response("respuesta completa", "stop", completion_tokens=30),
        infer_next=lambda _request: (_ for _ in ()).throw(
            AssertionError("stop must not continue")
        ),
    )

    assert result.content == "respuesta completa"
    assert result.metadata["model_request_count"] == 1
    assert result.metadata["auto_continuation_count"] == 0


def test_two_bounded_continuations_finish_on_second_request() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="detailed"))
    responses = iter(
        [
            _response(" dos", "length", completion_tokens=100),
            _response(" tres.", "stop", completion_tokens=100),
        ]
    )

    result = policy.complete_response(
        request,
        _response("Uno", "length", completion_tokens=100),
        infer_next=lambda _request: next(responses),
    )

    assert result.content == "Uno dos tres."
    assert result.metadata["auto_continuation_count"] == 2
    assert result.metadata["model_request_count"] == 3
    assert result.metadata["finish_reason"] == "stop"


def test_maximum_continuations_preserves_partial_and_length_state() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="detailed"))

    result = policy.complete_response(
        request,
        _response("A", "length", completion_tokens=100),
        infer_next=lambda _request: _response(
            "B", "length", completion_tokens=100
        ),
    )

    assert result.content == "ABB"
    assert result.metadata["auto_continuation_count"] == 2
    assert result.metadata["finish_reason"] == "length"
    assert result.metadata["segment_finish_reasons"] == [
        "length",
        "length",
        "length",
    ]
    assert result.metadata["continuation_stop_reason"] == (
        "max_continuations_reached"
    )


def test_context_pressure_blocks_continuation_without_losing_partial() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(
        ModelRequest(
            task_type="assistant",
            messages=[ModelMessage(role="user", content="x" * 1500)],
            metadata={
                "adaptive_output_budget": True,
                "auto_continue_text": True,
                "output_context_tokens": 512,
            },
            correlation_id="context-full",
        )
    )
    partial = "respuesta parcial"
    result = policy.complete_response(
        request,
        _response(partial, "length", completion_tokens=10),
        infer_next=lambda _request: (_ for _ in ()).throw(
            AssertionError("continuation must not run")
        ),
    )

    assert result.content == partial
    assert result.metadata["context_prevented_continuation"] is True
    assert result.metadata["continuation_stop_reason"] == "context_exhausted"
    assert result.metadata["auto_continuation_count"] == 0
    assert result.metadata["finish_reason"] == "length"


def test_total_response_budget_blocks_unbounded_fresh_segment_budget() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request())
    total_limit = policy.total_response_limit(request)

    result = policy.complete_response(
        request,
        _response(
            "respuesta extensa conservada",
            "length",
            completion_tokens=total_limit - 100,
        ),
        infer_next=lambda _request: (_ for _ in ()).throw(
            AssertionError("total response limit must prevent continuation")
        ),
    )

    assert result.content == "respuesta extensa conservada"
    assert result.metadata["continuation_stop_reason"] == "total_response_limit"
    assert result.metadata["total_response_token_limit"] == total_limit


def test_continuation_failure_preserves_partial_and_reports_safe_telemetry() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="detailed"))

    result = policy.complete_response(
        request,
        _response("contenido conservado", "length", completion_tokens=100),
        infer_next=lambda _request: (_ for _ in ()).throw(
            RuntimeError("private transport detail")
        ),
    )

    assert result.content == "contenido conservado"
    assert result.metadata["finish_reason"] == "length"
    assert result.metadata["auto_continuation_failed"] is True
    assert result.metadata["continuation_stop_reason"] == "continuation_failure"


def test_tool_calls_never_enter_textual_auto_continuation() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request())
    response = _response(
        "",
        "tool_calls",
        tool_calls=[ModelToolCall(id="one", name="files.search", arguments={})],
    )

    result = policy.complete_response(
        request,
        response,
        infer_next=lambda _request: (_ for _ in ()).throw(
            AssertionError("tool call must be executed by the agent")
        ),
    )

    assert result.tool_calls
    assert result.metadata["auto_continuation_count"] == 0


def test_no_overlap_keeps_both_segments_byte_for_byte() -> None:
    combined, removed = merge_continuation_text("```py\nprint(", "'ok')\n```")

    assert combined == "```py\nprint('ok')\n```"
    assert removed == 0


def test_continuation_merge_repairs_ellipsis_partial_word_overlap() -> None:
    combined, removed = merge_continuation_text(
        "La elección es crítica para la valide",
        "...para la validez de los resultados.",
    )

    assert combined == "La elección es crítica para la validez de los resultados."
    assert removed > 0

    exponential, exponential_removed = merge_continuation_text(
        "La capacidad ha crecido expon",
        "...exponencialmente durante la última década.",
    )
    assert exponential == (
        "La capacidad ha crecido exponencialmente durante la última década."
    )
    assert exponential_removed > 0


def test_continuation_merge_repairs_ellipsis_repeated_trailing_clause() -> None:
    combined, removed = merge_continuation_text(
        "La IA no solo es un conjunto de algoritmos, sino un paradigma",
        "...no solo es un conjunto de algoritmos, sino un paradigma que redefine la observación.",
    )

    assert combined == (
        "La IA no solo es un conjunto de algoritmos, sino un paradigma "
        "que redefine la observación."
    )
    assert removed > 0


def test_continuation_merge_repairs_restarted_clause_without_ellipsis() -> None:
    combined, removed = merge_continuation_text(
        "La inteligencia artificial (IA) ha dejado de ser una herramienta auxiliar para "
        "convertirse en un componente central y estructural",
        "La inteligencia artificial ha dejado de ser una herramienta auxiliar para "
        "convertirse en un componente central y estructural, actuando como núcleo científico.",
    )

    assert combined == (
        "La inteligencia artificial (IA) ha dejado de ser una herramienta auxiliar para "
        "convertirse en un componente central y estructural, actuando como núcleo científico."
    )
    assert removed > 0


def test_continuation_merge_collapses_short_word_and_restores_missing_space() -> None:
    repeated, repeated_removed = merge_continuation_text(
        "Es necesario asegurar que",
        "que las técnicas funcionen.",
    )
    joined, joined_removed = merge_continuation_text(
        "requiere validación",
        "humanas antes de confirmar.",
    )
    conjunction, conjunction_removed = merge_continuation_text(
        "según la información",
        "y el objetivo del análisis.",
    )
    short_words, short_words_removed = merge_continuation_text(
        "los humanos no",
        "ni siquiera pueden procesarlo.",
    )

    assert repeated == "Es necesario asegurar que las técnicas funcionen."
    assert repeated_removed == len("que")
    assert joined == "requiere validación humanas antes de confirmar."
    assert joined_removed == 0
    assert conjunction == "según la información y el objetivo del análisis."
    assert conjunction_removed == 0
    assert short_words == "los humanos no ni siquiera pueden procesarlo."
    assert short_words_removed == 0


def test_stream_continuations_remain_one_logical_stream() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="detailed"))
    calls = 0

    def stream_once(_request: ModelRequest) -> Iterator[StreamChunk]:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StreamChunk(text="Inicio repetido")
            yield StreamChunk(
                done=True,
                metadata={
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "completion_tokens": 50,
                },
            )
        else:
            yield StreamChunk(text="repetido final")
            yield StreamChunk(
                done=True,
                metadata={
                    "finish_reason": "stop",
                    "stream_closed_normally": True,
                    "completion_tokens": 50,
                },
            )

    chunks = list(policy.stream_completed(request, stream_once=stream_once))

    assert "".join(chunk.text for chunk in chunks) == "Inicio repetido final"
    assert sum(chunk.done for chunk in chunks) == 1
    assert chunks[-1].metadata["auto_continuation_count"] == 1
    assert chunks[-1].metadata["segment_finish_reasons"] == ["length", "stop"]


def test_cancelled_stream_never_starts_a_rolling_continuation() -> None:
    policy = OutputCompletionPolicy(_settings())
    request = policy.prepare_request(_request(profile="extended"))
    calls = 0

    def stream_once(_request: ModelRequest) -> Iterator[StreamChunk]:
        nonlocal calls
        calls += 1
        yield StreamChunk(text="Respuesta parcial")
        yield StreamChunk(
            done=True,
            metadata={
                "finish_reason": "cancelled",
                "stream_closed_normally": False,
                "incomplete": True,
            },
        )

    chunks = list(policy.stream_completed(request, stream_once=stream_once))

    assert calls == 1
    assert "".join(chunk.text for chunk in chunks) == "Respuesta parcial"
    assert chunks[-1].metadata["auto_continuation_count"] == 0
    assert chunks[-1].metadata["finish_reason"] == "cancelled"
