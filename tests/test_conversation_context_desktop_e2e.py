from __future__ import annotations

from threading import RLock

from jarvis.config import Settings
from jarvis.conversation_context import ConversationContextBuilder
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.desktop_runtime.base import DesktopChatMessage
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.models_runtime.base import ModelMessage, ModelRequest


class _CapturedModelResponse:
    def __init__(self, content: str = "Entendido.") -> None:
        self.content = content

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return {
            "content": self.content,
            "provider_name": "capture",
            "model_name": "nemotron-test",
            "latency_ms": 1.0,
        }


class _CaptureRuntime:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def infer_model(self, request: ModelRequest) -> _CapturedModelResponse:
        self.requests.append(request)
        return _CapturedModelResponse()


class _Bridge:
    def __init__(self, runtime: _CaptureRuntime) -> None:
        self.runtime = runtime


class _Panels:
    @staticmethod
    def compose():
        return None


def test_desktop_engine_preserves_opening_fact_after_twenty_turns() -> None:
    runtime = _CaptureRuntime()
    settings = Settings(
        _env_file=None,
        embeddings_enabled=False,
        voice_cleanup_enabled=False,
        conversation_context_max_tokens=4096,
    )
    engine = DesktopChatEngine(
        _Bridge(runtime),
        _Panels(),
        object(),
        settings,
        context_builder=ConversationContextBuilder(),
    )
    history: list[ModelMessage] = []

    def send(text: str) -> None:
        response = engine.handle(text, history=tuple(history))
        history.extend(
            [
                ModelMessage(role="user", content=text),
                ModelMessage(role="assistant", content=response.message.content),
            ]
        )

    send("Durante esta conversacion recuerda que la palabra es COBALTO-721.")
    for index in range(20):
        send(f"Mensaje de relleno {index}")
    send("¿Cual era la palabra?")

    final_request = runtime.requests[-1]
    assert any(
        message.role == "user" and "COBALTO-721" in message.content
        for message in final_request.messages
    )
    assert sum(message.role == "assistant" for message in final_request.messages) >= 20
    assert final_request.metadata["conversation_context"][
        "history_message_count"
    ] >= 40


def test_desktop_private_history_uses_enriched_user_context_and_excludes_banner() -> None:
    service = DesktopRuntimeService.__new__(DesktopRuntimeService)
    service._state_lock = RLock()  # noqa: SLF001
    service._conversation = [  # noqa: SLF001
        DesktopChatMessage(
            message_id="banner",
            role="system",
            content="JARVIS desktop interface online.",
        ),
        DesktopChatMessage(message_id="user-1", role="user", content="Analiza esto"),
        DesktopChatMessage(
            message_id="assistant-1", role="assistant", content="Veo una tabla."
        ),
    ]
    service._model_context_by_message_id = {  # noqa: SLF001
        "user-1": [
            ModelMessage(
                role="user",
                content="Analiza esto\n\nContenido derivado privado del adjunto",
            )
        ]
    }

    history = service._model_history_snapshot()  # noqa: SLF001

    assert [message.role for message in history] == ["user", "assistant"]
    assert "Contenido derivado privado" in history[0].content
    assert history[0].metadata["message_id"] == "user-1"
    assert all("interface online" not in message.content for message in history)


def test_streaming_request_uses_same_context_builder_as_normal_chat() -> None:
    runtime = _CaptureRuntime()
    settings = Settings(
        _env_file=None,
        embeddings_enabled=False,
        voice_cleanup_enabled=False,
        conversation_context_max_tokens=4096,
    )
    engine = DesktopChatEngine(
        _Bridge(runtime),
        _Panels(),
        object(),
        settings,
        context_builder=ConversationContextBuilder(),
    )
    history = [
        ModelMessage(
            role="user",
            content="Durante esta conversacion recuerda COBALTO-721.",
        ),
        ModelMessage(role="assistant", content="Entendido."),
        *[
            message
            for index in range(20)
            for message in (
                ModelMessage(role="user", content=f"relleno {index}"),
                ModelMessage(role="assistant", content="Entendido."),
            )
        ],
    ]

    prepared = engine.prepare_streaming_request("¿Cual era la palabra?")
    assert isinstance(prepared, tuple)
    request, _web_result = prepared
    finalized = engine.finalize_model_request(request, history=history)

    assert any("COBALTO-721" in message.content for message in finalized.messages)
    assert sum(message.role == "assistant" for message in finalized.messages) == 21
    assert finalized.metadata["conversation_context"][
        "history_message_count"
    ] == 42
