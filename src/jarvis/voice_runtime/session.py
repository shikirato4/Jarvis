from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from .base import AudioEvent, VoiceSession, VoiceSessionMode, VoiceSessionState


class VoiceSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._active_session_id: str | None = None
        self._lock = RLock()

    def start(self, session_id: str, *, correlation_id: str, mode: VoiceSessionMode, metadata: dict[str, object] | None = None) -> VoiceSession:
        session = VoiceSession(
            session_id=session_id,
            correlation_id=correlation_id,
            mode=mode,
            state=VoiceSessionState.LISTENING if mode != VoiceSessionMode.DICTATION else VoiceSessionState.DICTATING,
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session_id] = session
            self._active_session_id = session_id
        return session

    def get(self, session_id: str) -> VoiceSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def active(self) -> VoiceSession | None:
        with self._lock:
            return self._sessions.get(self._active_session_id) if self._active_session_id else None

    def update_state(self, session_id: str, state: VoiceSessionState, *, error: str | None = None) -> VoiceSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.state = state
            session.updated_at = datetime.now(timezone.utc)
            if error:
                session.last_error = error
            if state in {VoiceSessionState.STOPPED, VoiceSessionState.FAILED, VoiceSessionState.INTERRUPTED}:
                session.ended_at = datetime.now(timezone.utc)
                if self._active_session_id == session_id:
                    self._active_session_id = None
            return session

    def add_transcript(self, session_id: str, text: str) -> VoiceSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.transcripts.append(text)
            session.updated_at = datetime.now(timezone.utc)
            return session

    def add_event(self, session_id: str, event: AudioEvent) -> VoiceSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.events.append(event)
            session.updated_at = datetime.now(timezone.utc)
            return session

    def stop(self, session_id: str) -> VoiceSession | None:
        return self.update_state(session_id, VoiceSessionState.STOPPED)
