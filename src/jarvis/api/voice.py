from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.models.base import JarvisBaseModel
from jarvis.voice_runtime.base import VoiceSessionRequest


class VoiceTranscribeRequest(JarvisBaseModel):
    file_path: str


class VoiceSpeakRequest(JarvisBaseModel):
    text: str


class VoiceCancelRequest(JarvisBaseModel):
    correlation_id: str


def install_voice_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/voice/status")
    def voice_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).voice_runtime_service.status()

    @app.get("/voice/clap/status")
    def voice_clap_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).voice_runtime_service.clap_status()

    @app.get("/voice/session")
    def voice_session(request: Request) -> dict[str, Any]:
        session = get_jarvis(request).voice_runtime_service.active_session()
        return {"session": session.model_dump(mode="json") if session else None}

    @app.post("/voice/listen/start")
    def voice_listen_start(body: VoiceSessionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_start_session(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/voice/listen/stop")
    def voice_listen_stop(request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_stop_session().model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/voice/transcribe")
    def voice_transcribe(body: VoiceTranscribeRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_transcribe_file(body.file_path).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/voice/speak")
    def voice_speak(body: VoiceSpeakRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_speak(body.text).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/voice/dictate")
    def voice_dictate(body: VoiceSessionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_dictate(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/voice/cancel")
    def voice_cancel(body: VoiceCancelRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.voice_cancel(body.correlation_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
