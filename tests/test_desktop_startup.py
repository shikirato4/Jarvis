from __future__ import annotations

import jarvis.desktop as desktop_module


def test_desktop_main_shows_startup_surface_before_building_runtime(
    monkeypatch,
) -> None:
    events: list[str] = []

    class FakeQtApplication:
        def processEvents(self) -> None:
            events.append("process_events")

        def exec(self) -> int:
            events.append("event_loop")
            return 0

    class FakeWidget:
        def __init__(self, name: str) -> None:
            self.name = name

        def show(self) -> None:
            events.append(f"show_{self.name}")

        def close(self) -> None:
            events.append(f"close_{self.name}")

    class FakeBackend:
        def stop(self) -> None:
            events.append("backend_stop")

    class FakeDesktop:
        def start_backend_async(self) -> None:
            events.append("backend_async")

        def shutdown(self) -> None:
            events.append("desktop_shutdown")

    monkeypatch.setattr(desktop_module, "pyside_available", lambda: True)
    monkeypatch.setattr(
        desktop_module, "create_qt_application", lambda: FakeQtApplication()
    )
    monkeypatch.setattr(
        desktop_module,
        "_create_startup_indicator",
        lambda: FakeWidget("startup"),
        raising=False,
    )

    def _build(*, start=False):
        assert start is False
        events.append("build_runtime")
        return FakeBackend(), FakeDesktop()

    monkeypatch.setattr(desktop_module, "build_desktop_runtime", _build)
    monkeypatch.setattr(
        desktop_module,
        "JarvisDesktopWindow",
        lambda _desktop: FakeWidget("main"),
    )

    assert desktop_module.main() == 0
    assert events.index("show_startup") < events.index("build_runtime")
    assert events.index("process_events") < events.index("build_runtime")
    assert events.index("close_startup") < events.index("show_main")
