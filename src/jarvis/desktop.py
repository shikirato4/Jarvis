from __future__ import annotations

import signal


def pyside_available() -> bool:
    from jarvis.desktop_runtime.window import pyside_available as available

    return available()


def create_qt_application():
    from jarvis.desktop_runtime.window import create_qt_application as create

    return create()


def JarvisDesktopWindow(desktop):  # noqa: N802 - compatibility constructor.
    from jarvis.desktop_runtime.window import JarvisDesktopWindow as window_type

    return window_type(desktop)


def _install_signal_handlers(qt_app):
    if not hasattr(qt_app, "quit"):
        return None
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda _signum, _frame: qt_app.quit())
    return previous


def _restore_signal_handlers(previous) -> None:
    if previous is not None:
        signal.signal(signal.SIGINT, previous)


def build_desktop_runtime(settings=None, *, start: bool = True):
    from jarvis.bootstrap import build_application
    from jarvis.desktop_runtime.service import DesktopRuntimeService

    app = build_application(settings)
    desktop = DesktopRuntimeService(app)
    if start:
        app.start()
    return app, desktop


def _create_startup_indicator():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    surface = QWidget()
    surface.setObjectName("jarvisStartupSurface")
    surface.setWindowTitle("JARVIS")
    surface.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    surface.setFixedSize(440, 176)
    layout = QVBoxLayout(surface)
    layout.setContentsMargins(32, 28, 32, 28)
    title = QLabel("JARVIS")
    title.setObjectName("jarvisStartupTitle")
    status = QLabel("Inicializando servicios locales…")
    status.setObjectName("jarvisStartupStatus")
    status.setAccessibleName("Estado de inicio de JARVIS")
    layout.addWidget(title)
    layout.addStretch(1)
    layout.addWidget(status)
    surface.setStyleSheet(
        """
        QWidget#jarvisStartupSurface {
            background: #071015;
            border: 1px solid #25d9c8;
            border-radius: 12px;
        }
        QLabel#jarvisStartupTitle {
            color: #e7ffff;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 5px;
        }
        QLabel#jarvisStartupStatus {
            color: #7fb6b4;
            font-size: 13px;
        }
        """
    )
    return surface


def main() -> int:
    if not pyside_available():
        raise SystemExit("PySide6 is required to launch the JARVIS desktop app. Install it with `python -m pip install PySide6`.")

    qt_app = create_qt_application()
    previous_sigint = _install_signal_handlers(qt_app)
    startup = _create_startup_indicator()
    startup.show()
    qt_app.processEvents()
    backend = None
    desktop = None
    try:
        backend, desktop = build_desktop_runtime(start=False)
        desktop.start_backend_async()
        window = JarvisDesktopWindow(desktop)
        startup.close()
        window.show()
        return qt_app.exec()
    finally:
        _restore_signal_handlers(previous_sigint)
        startup.close()
        if desktop is not None:
            desktop.shutdown()
        if backend is not None:
            backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())
