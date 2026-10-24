from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.voice_runtime.gemini_live import GeminiLiveSession


class FakeTransport:
    """Scripted WebSocket double: records sent frames, plays server frames."""

    def __init__(self, server_frames: list[dict] | None = None) -> None:
        from collections import deque

        self.sent: list[str] = []
        self._queue: deque[dict] = deque(server_frames or [])
        self.connected = False
        self.closed = False
        self._lock = threading.Lock()

    def connect(self, url, headers=None) -> None:
        self.connected = True
        self.url = url

    def push_server_frame(self, frame: dict) -> None:
        with self._lock:
            self._queue.append(frame)

    def send(self, text: str) -> None:
        self.sent.append(text)

    def recv(self, timeout: float | None = None) -> str | None:
        with self._lock:
            if self._queue:
                return json.dumps(self._queue.popleft())
        if self.closed:
            return None
        time.sleep(0.01)
        return ""

    def close(self) -> None:
        self.closed = True
        self.connected = False


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "gemini_api_key": "live-key-1",
        "gemini_live_model": "gemini-2.0-flash-live-001",
        "gemini_live_voice": "Charon",
        "data_dir": tmp_path,
    }
    values.update(overrides)
    return Settings(**values)


def _audio_payload(text: bytes = b"\x01\x02\x03\x04") -> dict:
    return {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": base64.b64encode(text).decode(),
                        }
                    }
                ]
            }
        }
    }


def test_start_sends_setup_with_persona_model_and_voice(tmp_path: Path) -> None:
    transport = FakeTransport([{"setupComplete": {}}])
    session = GeminiLiveSession(_settings(tmp_path), transport_factory=lambda: transport)

    assert session.start().get("ready") is True

    setup = json.loads(transport.sent[0])["setup"]
    assert setup["model"] == "models/gemini-2.0-flash-live-001"
    assert (
        setup["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Charon"
    )
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    persona = " ".join(
        part["text"] for part in setup["systemInstruction"]["parts"]
    )
    assert "señor" in persona
    assert "JARVIS" in persona


def test_missing_api_key_raises_clear_error(tmp_path: Path) -> None:
    transport = FakeTransport()
    session = GeminiLiveSession(
        _settings(tmp_path, gemini_api_key=None), transport_factory=lambda: transport
    )

    with pytest.raises(JarvisError) as excinfo:
        session.start()

    assert "GEMINI_API_KEY" in str(excinfo.value)
    assert transport.connected is False


def test_send_audio_streams_base64_realtime_chunks(tmp_path: Path) -> None:
    transport = FakeTransport([{"setupComplete": {}}])
    session = GeminiLiveSession(_settings(tmp_path), transport_factory=lambda: transport)
    session.start()

    session.send_audio(b"\x00" * 5120)  # 160 ms @ 16 kHz 16-bit

    sent = [json.loads(frame) for frame in transport.sent[1:]]
    assert sent, "debe enviar chunks de realtimeInput"
    assert all("realtimeInput" in frame for frame in sent)
    decoded = b"".join(
        base64.b64decode(chunk["data"])
        for frame in sent
        for chunk in [frame["realtimeInput"]["audio"]]
    )
    assert decoded == b"\x00" * 5120
    for frame in sent:
        assert frame["realtimeInput"]["audio"]["mimeType"].startswith(
            "audio/pcm;rate=16000"
        )


def test_model_audio_routes_to_callback_and_turn_completes(tmp_path: Path) -> None:
    received: list[bytes] = []
    turns: list[bool] = []
    transport = FakeTransport(
        [
            {"setupComplete": {}},
            _audio_payload(b"aaaa"),
            _audio_payload(b"bbbb"),
            {"serverContent": {"turnComplete": True}},
        ]
    )
    session = GeminiLiveSession(
        _settings(tmp_path),
        transport_factory=lambda: transport,
        on_model_audio=received.append,
        on_turn_complete=lambda: turns.append(True),
    )
    session.start()

    deadline = time.monotonic() + 3
    while (len(received) < 2 or not turns) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert b"".join(received) == b"aaaabbbb"
    assert turns == [True]


def test_interruption_event_reaches_callback(tmp_path: Path) -> None:
    interrupted: list[bool] = []
    transport = FakeTransport(
        [{"setupComplete": {}}, {"serverContent": {"interrupted": True}}]
    )
    session = GeminiLiveSession(
        _settings(tmp_path),
        transport_factory=lambda: transport,
        on_interrupted=lambda: interrupted.append(True),
    )
    session.start()

    deadline = time.monotonic() + 3
    while not interrupted and time.monotonic() < deadline:
        time.sleep(0.02)

    assert interrupted == [True]


def test_stop_closes_transport_and_receiver(tmp_path: Path) -> None:
    transport = FakeTransport([{"setupComplete": {}}])
    session = GeminiLiveSession(_settings(tmp_path), transport_factory=lambda: transport)
    session.start()

    session.stop()

    assert transport.closed is True
    assert session.status()["active"] is False


# ---------------------------------------------- conversation service


def test_conversation_service_wires_mic_session_and_player(tmp_path: Path) -> None:
    from jarvis.voice_runtime.gemini_live import (
        GeminiLiveConversationService,
        GeminiLiveSession,
    )

    transport = FakeTransport([{"setupComplete": {}}])
    mic_chunks: list[bytes] = []
    played: list[bytes] = []

    class _FakeMic:
        def __init__(self, on_chunk) -> None:
            self._on_chunk = on_chunk
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _FakePlayer:
        def feed(self, pcm: bytes) -> None:
            played.append(pcm)

        def play_turn(self) -> None:
            return None

        def stop(self) -> None:
            return None

    service = GeminiLiveConversationService(
        _settings(tmp_path),
        session_factory=lambda: GeminiLiveSession(
            _settings(tmp_path),
            transport_factory=lambda: transport,
            on_model_audio=lambda pcm: None,
        ),
        mic_opener=lambda on_chunk: (
            mic_chunks.append(b"open"),
            _FakeMic(on_chunk),
        )[1],
        player_factory=_FakePlayer,
    )

    result = service.start_conversation()

    assert result["active"] is True
    assert b"open" in mic_chunks
    assert service.status()["active"] is True

    service.stop_conversation()
    assert service.status()["active"] is False


def test_conversation_service_requires_key(tmp_path: Path) -> None:
    from jarvis.voice_runtime.gemini_live import GeminiLiveConversationService
    from jarvis.voice_runtime.gemini_live import GeminiLiveSession

    service = GeminiLiveConversationService(
        _settings(tmp_path, gemini_api_key=None),
        session_factory=lambda: GeminiLiveSession(
            _settings(tmp_path, gemini_api_key=None)
        ),
    )

    with pytest.raises(JarvisError) as excinfo:
        service.start_conversation()

    assert "GEMINI_API_KEY" in str(excinfo.value)
