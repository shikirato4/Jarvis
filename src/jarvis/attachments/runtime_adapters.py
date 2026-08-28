from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

from jarvis.vision_runtime.base import OCRRequest


class MediaToolUnavailableError(RuntimeError):
    pass


class MediaToolExecutionError(RuntimeError):
    pass


class SubprocessMediaRunner:
    def run(self, args: list[str], *, timeout_seconds: float):
        try:
            completed = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaToolExecutionError(
                f"La herramienta multimedia no pudo completarse ({type(exc).__name__})."
            ) from exc
        if completed.returncode != 0:
            raise MediaToolExecutionError(
                f"La herramienta multimedia terminó con código {completed.returncode}."
            )
        return completed


class LocalMediaAdapters:
    def __init__(
        self,
        *,
        vision_service,
        voice_service,
        temp_root: str | Path,
        runner=None,
        ffmpeg_path: str | None = "auto",
        ffprobe_path: str | None = "auto",
        timeout_seconds: float = 120.0,
        max_video_frames: int = 12,
    ) -> None:
        self._vision = vision_service
        self._voice = voice_service
        self._temp_root = Path(temp_root).expanduser().resolve(strict=False)
        self._runner = runner or SubprocessMediaRunner()
        self._ffmpeg = shutil.which("ffmpeg") if ffmpeg_path == "auto" else ffmpeg_path
        self._ffprobe = (
            shutil.which("ffprobe") if ffprobe_path == "auto" else ffprobe_path
        )
        self._timeout_seconds = max(float(timeout_seconds), 1.0)
        self._max_video_frames = max(int(max_video_frames), 1)

    def ocr(self, data: bytes, mime_type: str) -> dict[str, Any]:
        receipt = self._vision.extract_text(
            OCRRequest(
                provider_name="tesseract_ocr",
                image_bytes=data,
                persist_result=False,
                metadata={
                    "source": "attachment",
                    "mime_type": mime_type,
                    "strict_provider": True,
                },
            )
        )
        blocks = []
        ocr_result = getattr(receipt, "ocr_result", None)
        for block in getattr(ocr_result, "blocks", []) or []:
            blocks.append(block.model_dump(mode="json"))
        return {
            "text": str(receipt.data.get("text") or ""),
            "language": receipt.data.get("language"),
            "blocks": blocks,
        }

    def transcribe(self, data: bytes, mime_type: str) -> dict[str, Any]:
        self._temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="audio-", dir=self._temp_root) as folder:
            workdir = Path(folder)
            input_path = workdir / f"input{_audio_suffix(mime_type)}"
            input_path.write_bytes(data)
            wav_path = input_path
            if mime_type != "audio/wav":
                self._require_ffmpeg()
                wav_path = workdir / "normalized.wav"
                self._runner.run(
                    [
                        str(self._ffmpeg),
                        "-v",
                        "error",
                        "-i",
                        str(input_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-y",
                        str(wav_path),
                    ],
                    timeout_seconds=self._timeout_seconds,
                )
            return self._transcribe_wav(wav_path)

    def analyze_video(self, data: bytes, mime_type: str) -> dict[str, Any]:
        self._require_video_tools()
        self._temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="video-", dir=self._temp_root) as folder:
            workdir = Path(folder)
            input_path = workdir / f"input{_video_suffix(mime_type)}"
            input_path.write_bytes(data)
            media_metadata = self._probe_metadata(input_path)
            duration = media_metadata["duration_seconds"]
            interval = video_sampling_interval(duration, self._max_video_frames)
            frame_pattern = workdir / "frame-%04d.jpg"
            self._runner.run(
                [
                    str(self._ffmpeg),
                    "-v",
                    "error",
                    "-i",
                    str(input_path),
                    "-vf",
                    f"fps=1/{interval:.6f}",
                    "-frames:v",
                    str(self._max_video_frames),
                    "-q:v",
                    "3",
                    "-y",
                    str(frame_pattern),
                ],
                timeout_seconds=self._timeout_seconds,
            )
            frames = []
            for index, frame_path in enumerate(sorted(workdir.glob("frame-*.jpg"))):
                ocr = self.ocr(frame_path.read_bytes(), "image/jpeg")
                frames.append(
                    {
                        "timestamp_seconds": round(index * interval, 3),
                        "description": str(
                            ocr.get("text") or "Fotograma sin texto OCR"
                        ),
                        "ocr": ocr,
                    }
                )
            audio_path = workdir / "audio.wav"
            transcript: list[dict[str, Any]] = []
            audio_error: str | None = None
            try:
                self._runner.run(
                    [
                        str(self._ffmpeg),
                        "-v",
                        "error",
                        "-i",
                        str(input_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-y",
                        str(audio_path),
                    ],
                    timeout_seconds=self._timeout_seconds,
                )
                transcript = list(
                    self._transcribe_wav(audio_path).get("segments") or []
                )
            except MediaToolExecutionError as exc:
                audio_error = str(exc)
            summary_parts = [
                frame["description"] for frame in frames if frame["description"]
            ]
            summary_parts.extend(
                str(segment.get("text") or "") for segment in transcript
            )
            return {
                "duration_seconds": duration,
                "metadata": media_metadata,
                "frames": frames,
                "transcript": transcript,
                "summary": "\n".join(part for part in summary_parts if part),
                "audio_error": audio_error,
            }

    def _transcribe_wav(self, path: Path) -> dict[str, Any]:
        receipt = self._voice.transcribe_file(str(path))
        return {
            "text": str(receipt.data.get("text") or ""),
            "segments": list(receipt.data.get("segments") or []),
            "provider": receipt.data.get("provider"),
            "backend": receipt.data.get("backend"),
        }

    def _probe_metadata(self, path: Path) -> dict[str, Any]:
        completed = self._runner.run(
            [
                str(self._ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=codec_type,codec_name,width,height,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            timeout_seconds=self._timeout_seconds,
        )
        return parse_media_probe(completed.stdout)

    def _probe_duration(self, path: Path) -> float:
        return float(self._probe_metadata(path)["duration_seconds"])

    def _require_ffmpeg(self) -> None:
        if not self._ffmpeg:
            raise MediaToolUnavailableError(
                "FFmpeg no está disponible; se necesita para normalizar este formato de audio."
            )

    def _require_video_tools(self) -> None:
        missing = []
        if not self._ffmpeg:
            missing.append("FFmpeg")
        if not self._ffprobe:
            missing.append("FFprobe")
        if missing:
            raise MediaToolUnavailableError(
                f"{' y '.join(missing)} no están disponibles para analizar video localmente."
            )


def _audio_suffix(mime_type: str) -> str:
    return {
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/flac": ".flac",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
    }.get(mime_type, ".audio")


def _video_suffix(mime_type: str) -> str:
    return {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
    }.get(mime_type, ".video")


def parse_media_probe(raw_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaToolExecutionError(
            "FFprobe no devolvió una duración válida."
        ) from exc
    if duration <= 0:
        raise MediaToolExecutionError("FFprobe devolvió una duración no positiva.")
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
        None,
    )
    metadata: dict[str, Any] = {
        "duration_seconds": duration,
        "format": str(payload.get("format", {}).get("format_name") or "unknown"),
        "video": None,
        "audio": None,
    }
    if video_stream is not None:
        metadata["video"] = {
            "codec": str(video_stream.get("codec_name") or "unknown"),
            "width": _optional_int(video_stream.get("width")),
            "height": _optional_int(video_stream.get("height")),
        }
    if audio_stream is not None:
        metadata["audio"] = {
            "codec": str(audio_stream.get("codec_name") or "unknown"),
            "sample_rate": _optional_int(audio_stream.get("sample_rate")),
            "channels": _optional_int(audio_stream.get("channels")),
        }
    return metadata


def video_sampling_interval(duration_seconds: float, max_frames: int) -> float:
    if duration_seconds <= 0:
        raise ValueError("video duration must be positive")
    return max(float(duration_seconds) / max(int(max_frames), 1), 0.04)


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
