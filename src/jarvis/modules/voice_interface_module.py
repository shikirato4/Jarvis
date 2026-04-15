from __future__ import annotations

from pydantic import BaseModel

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.tools.models import ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry
from jarvis.voice_runtime.base import VoiceSessionRequest
from jarvis.voice_runtime.service import VoiceRuntimeService


class VoiceSpeakPayload(BaseModel):
    text: str


class VoiceTranscribePayload(BaseModel):
    file_path: str


class VoiceListenPayload(BaseModel):
    duration_seconds: float | None = None
    language: str | None = None
    playback_response: bool = False


class VoiceDictationPayload(BaseModel):
    duration_seconds: float | None = None
    language: str | None = None
    target_window: str | None = None
    ui_mode: str | None = None


class VoiceCancelPayload(BaseModel):
    correlation_id: str


class VoiceInterfaceModule:
    name = "voice_interface"
    description = "Live voice control, transcription, synthesis and dictation integration."

    def __init__(self, voice_runtime: VoiceRuntimeService) -> None:
        self._voice = voice_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="voice_runtime.speak", description="Speak text through the configured TTS pipeline.", payload_model=VoiceSpeakPayload, handler=self._speak, tags=("voice", "tts")))
        registry.register(ActionDefinition(name="voice_runtime.transcribe_file", description="Transcribe audio from a file through the configured STT pipeline.", payload_model=VoiceTranscribePayload, handler=self._transcribe_file, tags=("voice", "stt")))
        registry.register(ActionDefinition(name="voice_runtime.listen_start", description="Start an active voice listening session.", payload_model=VoiceListenPayload, handler=self._listen_start, tags=("voice", "session")))
        registry.register(ActionDefinition(name="voice_runtime.dictate", description="Start a dictation session and deliver text to the configured sink.", payload_model=VoiceDictationPayload, handler=self._dictate, tags=("voice", "dictation")))
        registry.register(ActionDefinition(name="voice_runtime.cancel", description="Cancel an active voice operation.", payload_model=VoiceCancelPayload, handler=self._cancel, tags=("voice", "control")))

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(ToolDefinition(name="voice.status", description="Inspect the current voice runtime state.", input_model=VoiceListenPayload, handler=self._tool_status, tags=("voice", "state")))
        registry.register(ToolDefinition(name="voice.speak", description="Speak text through the configured TTS pipeline.", input_model=VoiceSpeakPayload, handler=self._tool_speak, tags=("voice", "tts")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="voice.runtime_control",
                module_name=self.name,
                intent="voice_runtime",
                description="Control the live voice runtime, transcription and speech output.",
                action_names=("voice_runtime.speak", "voice_runtime.transcribe_file", "voice_runtime.listen_start", "voice_runtime.dictate", "voice_runtime.cancel"),
                tool_names=("voice.status", "voice.speak"),
                keywords=("escucha", "listen", "dictado", "dictate", "transcribe", "lee en voz alta", "speak aloud"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _speak(self, context: ActionContext, payload: VoiceSpeakPayload) -> ActionResult:
        receipt = self._voice.speak(payload.text, correlation_id=context.correlation_id)
        return ActionResult(message="voice synthesis completed", data=receipt.model_dump(mode="json"))

    def _transcribe_file(self, context: ActionContext, payload: VoiceTranscribePayload) -> ActionResult:
        receipt = self._voice.transcribe_file(payload.file_path, correlation_id=context.correlation_id)
        return ActionResult(message="audio transcribed", data=receipt.model_dump(mode="json"))

    def _listen_start(self, context: ActionContext, payload: VoiceListenPayload) -> ActionResult:
        receipt = self._voice.start_session(
            VoiceSessionRequest(
                duration_seconds=payload.duration_seconds,
                language=payload.language,
                playback_response=payload.playback_response,
                correlation_id=context.correlation_id,
            )
        )
        return ActionResult(message="voice listening started", data=receipt.model_dump(mode="json"))

    def _dictate(self, context: ActionContext, payload: VoiceDictationPayload) -> ActionResult:
        receipt = self._voice.dictate_once(
            VoiceSessionRequest(
                mode="dictation",
                duration_seconds=payload.duration_seconds,
                language=payload.language,
                target_window=payload.target_window,
                ui_mode=payload.ui_mode,
                correlation_id=context.correlation_id,
            )
        )
        return ActionResult(message="voice dictation started", data=receipt.model_dump(mode="json"))

    def _cancel(self, context: ActionContext, payload: VoiceCancelPayload) -> ActionResult:
        receipt = self._voice.cancel(payload.correlation_id)
        return ActionResult(message="voice cancellation requested", data=receipt.model_dump(mode="json"))

    def _tool_status(self, context: ToolContext, payload: VoiceListenPayload) -> ToolResult:
        return ToolResult(message="voice status retrieved", data=self._voice.status())

    def _tool_speak(self, context: ToolContext, payload: VoiceSpeakPayload) -> ToolResult:
        receipt = self._voice.speak(payload.text, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if payload.get("file_path"):
            return [ActionStep(action="voice_runtime.transcribe_file", payload=payload)]
        if payload.get("target_window") or payload.get("ui_mode"):
            return [ActionStep(action="voice_runtime.dictate", payload=payload)]
        if request.query:
            payload.setdefault("text", request.query)
            return [ActionStep(action="voice_runtime.speak", payload=payload)]
        return [ActionStep(action="voice_runtime.listen_start", payload=payload)]
