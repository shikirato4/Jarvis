from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.attachments.runtime_adapters import (
    LocalMediaAdapters,
    MediaToolUnavailableError,
)


class FakeVision:
    def extract_text(self, request):
        assert request.image_bytes
        return SimpleNamespace(
            data={"text": "visible text", "language": "es"},
            ocr_result=SimpleNamespace(
                blocks=[
                    SimpleNamespace(model_dump=lambda mode: {"text": "visible text"})
                ]
            ),
        )


class FakeVoice:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe_file(self, file_path: str):
        path = Path(file_path)
        assert path.exists()
        self.paths.append(path)
        return SimpleNamespace(
            data={
                "text": "spoken text",
                "provider": "fake-stt",
                "backend": "wav_file",
                "segments": [
                    {"text": "spoken text", "start_seconds": 0.0, "end_seconds": 1.25}
                ],
            }
        )


class FakeMediaRunner:
    def run(self, args: list[str], *, timeout_seconds: float):
        assert timeout_seconds > 0
        if "ffprobe" in Path(args[0]).name:
            return SimpleNamespace(stdout=json.dumps({"format": {"duration": "4.0"}}))
        output = Path(args[-1])
        if "%04d" in output.name:
            output.with_name("frame-0001.jpg").write_bytes(b"jpeg-one")
            output.with_name("frame-0002.jpg").write_bytes(b"jpeg-two")
        else:
            output.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        return SimpleNamespace(stdout="")


def test_ocr_adapter_returns_text_and_blocks_without_local_path(tmp_path) -> None:
    adapters = LocalMediaAdapters(
        vision_service=FakeVision(),
        voice_service=FakeVoice(),
        temp_root=tmp_path,
        ffmpeg_path=None,
        ffprobe_path=None,
    )

    result = adapters.ocr(b"image", "image/png")

    assert result["text"] == "visible text"
    assert result["blocks"] == [{"text": "visible text"}]
    assert "path" not in str(result).casefold()


def test_audio_adapter_converts_non_wav_and_cleans_temporary_files(tmp_path) -> None:
    voice = FakeVoice()
    adapters = LocalMediaAdapters(
        vision_service=FakeVision(),
        voice_service=voice,
        temp_root=tmp_path,
        runner=FakeMediaRunner(),
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
    )

    result = adapters.transcribe(b"ID3audio", "audio/mpeg")

    assert result["segments"][0]["end_seconds"] == 1.25
    assert result["provider"] == "fake-stt"
    assert all(not path.exists() for path in voice.paths)


def test_video_adapter_extracts_frames_audio_timestamps_and_cleans_files(
    tmp_path,
) -> None:
    voice = FakeVoice()
    adapters = LocalMediaAdapters(
        vision_service=FakeVision(),
        voice_service=voice,
        temp_root=tmp_path,
        runner=FakeMediaRunner(),
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        max_video_frames=2,
    )

    result = adapters.analyze_video(b"video", "video/mp4")

    assert [frame["timestamp_seconds"] for frame in result["frames"]] == [0.0, 2.0]
    assert result["frames"][0]["description"] == "visible text"
    assert result["transcript"][0]["text"] == "spoken text"
    assert result["duration_seconds"] == 4.0
    assert all(not path.exists() for path in voice.paths)
    assert list(tmp_path.iterdir()) == []


def test_audio_and_video_report_exact_missing_media_tool(tmp_path) -> None:
    adapters = LocalMediaAdapters(
        vision_service=FakeVision(),
        voice_service=FakeVoice(),
        temp_root=tmp_path,
        ffmpeg_path=None,
        ffprobe_path=None,
    )

    with pytest.raises(MediaToolUnavailableError, match="FFmpeg"):
        adapters.transcribe(b"ID3audio", "audio/mpeg")
    with pytest.raises(MediaToolUnavailableError, match="FFmpeg.*FFprobe"):
        adapters.analyze_video(b"video", "video/mp4")
