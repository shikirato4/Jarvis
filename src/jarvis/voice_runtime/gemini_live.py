from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from jarvis.config import Settings
from jarvis.core.errors import JarvisError

logger = logging.getLogger("jarvis.voice.live")

_LIVE_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Persona del mayordomo para la sesion Live: corta, clara, en espanol.
_LIVE_PERSONA = (
    "Eres JARVIS, el mayordomo personal del usuario. Le hablas en espanol, "
    "le dices 'señor', respondes breve y util, con personalidad educada y "
    "unalgo ingeniosa. Si te piden operar la computadora, describe el paso "
    "que haras. No inventes hechos: si no sabes algo, dilo."
)

_INPUT_SAMPLE_RATE = 16_000
_INPUT_BYTES_PER_MS = 2 * 16  # 16-bit mono -> 32 bytes per ms
_CHUNK_MS = 100


class LiveTransport(Protocol):
    def connect(self, url: str, headers: dict | None = None) -> None: ...

    def send(self, text: str) -> None: ...

    def recv(self, timeout: float | None = None) -> str | None: ...

    def close(self) -> None: ...


class WebSocketTransport:
    """websocket-client (sync) transport for the Live API."""

    def __init__(self) -> None:
        self._ws = None

    def connect(self, url: str, headers: dict | None = None) -> None:
        import websocket  # websocket-client

        self._ws = websocket.create_connection(url, header=headers or [], timeout=10)

    def send(self, text: str) -> None:
        if self._ws is None:
            raise JarvisError(
                "Live transport sin conexion",
                component="voice_runtime",
                code="live_not_connected",
                recoverable=True,
            )
        self._ws.send(text)

    def recv(self, timeout: float | None = None) -> str | None:
        if self._ws is None:
            return None
        try:
            self._ws.settimeout(timeout or 10)
            return self._ws.recv()
        except Exception as exc:  # noqa: BLE001
            # websocket-client raises its own WebSocketTimeoutException
            # (not TimeoutError) when the socket read times out: that is
            # "keep waiting", not "connection closed".
            if type(exc).__name__ == "WebSocketTimeoutException":
                return ""
            return None

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None


class GeminiLiveSession:
    """One bidirectional voice conversation with the Gemini Live API.

    Mic audio (16 kHz mono s16le PCM) goes up as base64 realtimeInput
    chunks; the model's turn comes back as 24 kHz PCM parts delivered
    through `on_model_audio`, with turn/interruption events for the
    caller to manage playback.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport_factory: Callable[[], LiveTransport] | None = None,
        on_model_audio: Callable[[bytes], None] | None = None,
        on_turn_complete: Callable[[], None] | None = None,
        on_interrupted: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._transport_factory = transport_factory
        self._on_model_audio = on_model_audio
        self._on_turn_complete = on_turn_complete
        self._on_interrupted = on_interrupted
        self._on_error = on_error
        self._transport: LiveTransport | None = None
        self._receiver: threading.Thread | None = None
        self._active = threading.Event()
        self._ready = threading.Event()
        self._send_lock = threading.Lock()
        self._last_errors: list[str] = []
        self._error_lock = threading.Lock()

        def _capture_error(message: str) -> None:
            with self._error_lock:
                self._last_errors.append(message)
            if on_error:
                on_error(message)

        self._error_sink = _capture_error
        self._on_error = _capture_error

    def _api_key(self) -> str:
        raw = getattr(self._settings, "gemini_api_key", None)
        value = raw.get_secret_value() if raw is not None else ""
        if not value:
            raise JarvisError(
                "Gemini Live necesita GEMINI_API_KEY en .env "
                "(gratis en aistudio.google.com, sin tarjeta)",
                component="voice_runtime",
                code="live_not_configured",
                recoverable=True,
            )
        return value

    def start(self) -> dict[str, Any]:
        key = self._api_key()
        transport = (
            self._transport_factory() if self._transport_factory else WebSocketTransport()
        )
        url = f"{_LIVE_WS_BASE}?key={key}"
        transport.connect(url)
        self._transport = transport
        self._active.set()
        self._ready.clear()

        setup = {
            "setup": {
                "model": f"models/{self._settings.gemini_live_model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self._settings.gemini_live_voice,
                            }
                        }
                    },
                },
                "systemInstruction": {"parts": [{"text": _LIVE_PERSONA}]},
            }
        }
        transport.send(json.dumps(setup))

        self._receiver = threading.Thread(
            target=self._receive_loop, name="jarvis-live-receiver", daemon=True
        )
        self._receiver.start()

        deadline = time.monotonic() + 10
        while not self._ready.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._ready.is_set():
            errors_seen = list(self._last_errors)
            self.stop()
            raise JarvisError(
                "Gemini Live no confirmo la sesion (setupComplete)"
                + (f"; servidor: {errors_seen[-1]}" if errors_seen else ""),
                component="voice_runtime",
                code="live_handshake_timeout",
                recoverable=True,
            )
        logger.info(
            "gemini_live_started",
            extra={
                "model": self._settings.gemini_live_model,
                "voice": self._settings.gemini_live_voice,
            },
        )
        return {"ready": True, "model": self._settings.gemini_live_model}

    def send_audio(self, pcm: bytes) -> None:
        """Stream mic PCM (16 kHz mono s16le) in ~100 ms base64 chunks."""
        if not self._active.is_set() or self._transport is None:
            return
        chunk_bytes = _CHUNK_MS * _INPUT_BYTES_PER_MS
        with self._send_lock:
            for offset in range(0, len(pcm), chunk_bytes):
                piece = pcm[offset : offset + chunk_bytes]
                frame = {
                    "realtimeInput": {
                        "audio": {
                            "mimeType": f"audio/pcm;rate={_INPUT_SAMPLE_RATE}",
                            "data": base64.b64encode(piece).decode(),
                        }
                    }
                }
                try:
                    self._transport.send(json.dumps(frame))
                except Exception as exc:  # noqa: BLE001 - surface, keep session
                    if self._on_error:
                        self._on_error(f"envio de audio fallo: {exc}")

    def _receive_loop(self) -> None:
        while self._active.is_set():
            transport = self._transport
            if transport is None:
                return
            try:
                raw = transport.recv(timeout=0.25)
            except Exception:  # noqa: BLE001
                raw = None
            if isinstance(raw, bytes):
                # The Live API sends every frame as a BINARY payload
                # containing JSON text; protocol errors are also binary
                # (\x03\xf0 + message). Decode and route accordingly.
                try:
                    decoded = raw.decode("utf-8", errors="strict")
                    json.loads(decoded)
                    raw = decoded
                except (UnicodeDecodeError, ValueError):
                    if self._on_error and raw:
                        message = raw.decode("utf-8", errors="replace").lstrip("\x03\xf0")
                        self._on_error(f"servidor (binario): {message}")
                    self._active.clear()
                    return
            if raw in (None, ""):
                if raw is None and transport is self._transport:
                    # connection closed by the server
                    if self._active.is_set() and self._on_error:
                        self._on_error("conexion cerrada por el servidor")
                    self._active.clear()
                continue
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            self._handle_frame(frame)

    def _handle_frame(self, frame: dict) -> None:
        if "setupComplete" in frame:
            self._ready.set()
            return
        if "error" in frame:
            detail = frame["error"]
            if self._on_error:
                self._on_error(f"servidor: {detail}")
            self._active.clear()
            return
        server = frame.get("serverContent") or {}
        if server.get("interrupted"):
            if self._on_interrupted:
                self._on_interrupted()
            return
        turn = server.get("modelTurn") or {}
        for part in turn.get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                try:
                    pcm = base64.b64decode(inline["data"])
                except Exception:  # noqa: BLE001
                    continue
                if self._on_model_audio:
                    self._on_model_audio(pcm)
        if server.get("turnComplete"):
            if self._on_turn_complete:
                self._on_turn_complete()
        if "goAway" in frame:
            if self._on_error:
                self._on_error(
                    "el servidor aviso que la sesion termina pronto (goAway)"
                )

    def stop(self) -> None:
        self._active.clear()
        transport = self._transport
        self._transport = None
        if transport is not None:
            transport.close()
        receiver = self._receiver
        if receiver is not None and receiver.is_alive() and receiver is not threading.current_thread():
            receiver.join(timeout=2)

    def status(self) -> dict[str, Any]:
        return {
            "active": self._active.is_set() and self._transport is not None,
            "ready": self._ready.is_set(),
            "model": self._settings.gemini_live_model,
            "voice": self._settings.gemini_live_voice,
        }


class SoundDeviceMicStream:
    """Continuous 16 kHz mono capture feeding the Live session."""

    def __init__(
        self,
        on_chunk: Callable[[bytes], None],
        *,
        sample_rate: int = _INPUT_SAMPLE_RATE,
        block_ms: int = 100,
    ) -> None:
        import numpy  # noqa: F401 - sounddevice needs it
        import sounddevice

        frames = int(sample_rate * block_ms / 1000)

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            on_chunk(bytes(indata))

        self._stream = sounddevice.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frames,
            callback=_callback,
        )
        self._stream.start()

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


class WinsoundTurnPlayer:
    """Accumulates a model turn (24 kHz PCM) and plays it via winsound."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm)

    def play_turn(self) -> None:
        with self._lock:
            pending = bytes(self._buffer)
            self._buffer.clear()
        if not pending:
            return

        def _play() -> None:
            import tempfile
            from pathlib import Path

            import winsound

            from jarvis.voice_runtime.backend_elevenlabs import _pcm_to_wav

            wav = _pcm_to_wav(pending, sample_rate=24_000)
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - handle outlives block: winsound plays async to this thread
                prefix="jarvis-live-", suffix=".wav", delete=False
            )
            handle.write(wav)
            handle.close()
            try:
                winsound.PlaySound(handle.name, winsound.SND_FILENAME | winsound.SND_SYNC)
            finally:
                Path(handle.name).unlink(missing_ok=True)

        self.stop()
        self._thread = threading.Thread(target=_play, name="jarvis-live-play", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:  # noqa: BLE001
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


class GeminiLiveConversationService:
    """Owns one live conversation: mic stream -> Live session -> speaker."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_factory: Callable[[], GeminiLiveSession] | None = None,
        mic_opener: Callable[[Callable[[bytes], None]], Any] | None = None,
        player_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._mic_opener = mic_opener
        self._player_factory = player_factory
        self._session: GeminiLiveSession | None = None
        self._mic: Any = None
        self._player: Any = None
        self._lock = threading.Lock()

    def _new_session(self) -> GeminiLiveSession:
        if self._session_factory is not None:
            return self._session_factory()
        return GeminiLiveSession(self._settings)

    def start_conversation(self) -> dict[str, Any]:
        with self._lock:
            self._stop_locked()
            player = (self._player_factory or WinsoundTurnPlayer)()
            self._player = player
            session = self._new_session()
            # Wire the service callbacks whatever the factory built.
            session._on_model_audio = player.feed  # noqa: SLF001
            session._on_turn_complete = player.play_turn  # noqa: SLF001
            session._on_interrupted = player.stop  # noqa: SLF001
            result = session.start()
            self._session = session
            open_mic = self._mic_opener or SoundDeviceMicStream
            self._mic = open_mic(session.send_audio)
            logger.info(
                "gemini_live_conversation_started",
                extra={"model": self._settings.gemini_live_model},
            )
            return {"active": True, **result}

    def _stop_locked(self) -> None:
        mic = self._mic
        self._mic = None
        if mic is not None:
            try:
                mic.close()
            except Exception:  # noqa: BLE001
                pass
        session = self._session
        self._session = None
        if session is not None:
            session.stop()
        player = self._player
        self._player = None
        if player is not None:
            try:
                player.stop()
            except Exception:  # noqa: BLE001
                pass

    def stop_conversation(self) -> dict[str, Any]:
        with self._lock:
            was_active = self._session is not None
            self._stop_locked()
        if was_active:
            logger.info("gemini_live_conversation_stopped")
        return {"active": False}

    def status(self) -> dict[str, Any]:
        session = self._session
        return {
            "active": session is not None,
            "model": self._settings.gemini_live_model,
            "voice": self._settings.gemini_live_voice,
        }
