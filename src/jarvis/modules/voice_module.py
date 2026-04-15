from __future__ import annotations

import mimetypes
import wave

from pydantic import BaseModel

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.safety import ensure_within_roots


class InspectAudioPayload(BaseModel):
    audio_path: str


class VoiceModule:
    name = "voice"
    description = "Audio intake and local metadata inspection."

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="voice.inspect_audio",
                description="Inspect local audio metadata, with deep support for WAV files.",
                payload_model=InspectAudioPayload,
                handler=self._inspect_audio,
                tags=("voice", "inspection"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="voice.inspect",
                module_name=self.name,
                intent="voice",
                description="Inspect local audio input.",
                action_names=("voice.inspect_audio",),
                tool_names=("audio.inspect",),
                keywords=("audio", "voz", "voice", "wav"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _inspect_audio(self, context: ActionContext, payload: InspectAudioPayload) -> ActionResult:
        allowed_roots = (context.settings.resolved_workspace_root, *context.settings.resolved_research_roots)
        audio_path = ensure_within_roots(payload.audio_path, allowed_roots, "audio inspection")
        file_size = audio_path.stat().st_size
        data = {
            "path": str(audio_path),
            "suffix": audio_path.suffix.lower(),
            "mime_type": mimetypes.guess_type(audio_path.name)[0],
            "size_bytes": file_size,
        }
        if audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as audio_file:
                frame_rate = audio_file.getframerate()
                frame_count = audio_file.getnframes()
                data.update(
                    {
                        "channels": audio_file.getnchannels(),
                        "sample_width_bytes": audio_file.getsampwidth(),
                        "frame_rate": frame_rate,
                        "duration_seconds": round(frame_count / frame_rate, 3) if frame_rate else 0,
                    }
                )
        else:
            data["inspection_note"] = "Deep waveform inspection currently supports WAV files."
        return ActionResult(message="audio inspected", data=data)

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        return [ActionStep(action="voice.inspect_audio", payload=dict(request.payload))]
