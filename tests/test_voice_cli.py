from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from jarvis.cli import app


def test_voice_speak_cli_waits_for_completion_before_shutdown(monkeypatch) -> None:
    calls: list[object] = []

    class _Receipt:
        correlation_id = "speech-1"
        data = {"queued": True}

        def model_dump(self, **_kwargs):
            return {"correlation_id": self.correlation_id, "data": {"queued": True}}

    runtime = SimpleNamespace(
        voice_speak=lambda text: calls.append(("speak", text)) or _Receipt(),
        voice_wait_for_speech=lambda correlation_id, timeout_seconds: calls.append(
            ("wait", correlation_id, timeout_seconds)
        )
        or {"status": "completed", "provider": "windows_sapi"},
    )

    class _Jarvis:
        runtime_service = runtime
        settings = SimpleNamespace(voice_watchdog_timeout_ms=45_000)

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr("jarvis.cli.build_application", _Jarvis)

    result = CliRunner().invoke(app, ["voice", "speak", "hola"])

    assert result.exit_code == 0
    assert calls == [
        "start",
        ("speak", "hola"),
        ("wait", "speech-1", 45.0),
        "stop",
    ]
    assert '"status": "completed"' in result.stdout
