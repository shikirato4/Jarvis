from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.desktop_runtime import DesktopRuntimeService, JarvisDesktopWindow, create_qt_application, pyside_available


def build_desktop_runtime(settings=None, *, start: bool = True):
    app = build_application(settings)
    desktop = DesktopRuntimeService(app)
    if start:
        app.start()
    return app, desktop


def main() -> int:
    if not pyside_available():
        raise SystemExit("PySide6 is required to launch the JARVIS desktop app. Install it with `python -m pip install PySide6`.")
    backend, desktop = build_desktop_runtime(start=False)
    desktop.start_backend_async()
    qt_app = create_qt_application()
    window = JarvisDesktopWindow(desktop)
    window.show()
    try:
        return qt_app.exec()
    finally:
        desktop.shutdown()
        backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())
