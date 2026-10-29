from __future__ import annotations

import base64
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.image_runtime.gemini_backend import GeminiImageProvider


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeHttp:
    """Records the request and returns a queued response."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, *, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"fake-pixel-data" * 8


def _ok_payload() -> dict:
    data = base64.b64encode(_png_bytes()).decode()
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "aqui esta tu imagen"},
                        {"inlineData": {"mimeType": "image/png", "data": data}},
                    ]
                }
            }
        ]
    }


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "gemini_api_key": "test-key-123",
        "gemini_image_model": "gemini-2.5-flash-image",
        "data_dir": tmp_path / "runtime",
    }
    values.update(overrides)
    return Settings(**values)


def test_generates_saves_png_and_returns_path(tmp_path: Path) -> None:
    http = _FakeHttp(_FakeResponse(200, _ok_payload()))
    provider = GeminiImageProvider(_settings(tmp_path), http=http)

    path = provider.generate("un gato astronauta", output_dir=tmp_path / "out")

    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert path.suffix == ".png"
    request = http.calls[0]
    assert request["url"].endswith(
        "models/gemini-2.5-flash-image:generateContent"
    )
    assert request["headers"]["x-goog-api-key"] == "test-key-123"
    assert request["json"]["contents"][0]["parts"][0]["text"] == "un gato astronauta"
    assert "IMAGE" in request["json"]["generationConfig"]["responseModalities"]


def test_missing_key_raises_clear_error(tmp_path: Path) -> None:
    provider = GeminiImageProvider(
        _settings(tmp_path, gemini_api_key=None), http=_FakeHttp(_FakeResponse(200, _ok_payload()))
    )

    with pytest.raises(JarvisError) as excinfo:
        provider.generate("algo", output_dir=tmp_path)

    assert "GEMINI_API_KEY" in str(excinfo.value)


def test_http_error_maps_to_jarvis_error(tmp_path: Path) -> None:
    http = _FakeHttp(_FakeResponse(429, {"error": {"message": "quota"}}))
    provider = GeminiImageProvider(_settings(tmp_path), http=http)

    with pytest.raises(JarvisError) as excinfo:
        provider.generate("algo", output_dir=tmp_path)

    assert "429" in str(excinfo.value) or "quota" in str(excinfo.value)


def test_response_without_image_parts_raises(tmp_path: Path) -> None:
    http = _FakeHttp(
        _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "solo texto"}]}}]})
    )
    provider = GeminiImageProvider(_settings(tmp_path), http=http)

    with pytest.raises(JarvisError) as excinfo:
        provider.generate("algo", output_dir=tmp_path)

    assert "imagen" in str(excinfo.value).casefold()
