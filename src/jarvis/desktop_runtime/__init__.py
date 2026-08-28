__all__ = ["DesktopRuntimeService", "JarvisDesktopWindow", "create_qt_application", "pyside_available"]


def __getattr__(name: str):
    if name == "DesktopRuntimeService":
        from .service import DesktopRuntimeService

        return DesktopRuntimeService
    if name in {"JarvisDesktopWindow", "create_qt_application", "pyside_available"}:
        from . import window

        return getattr(window, name)
    raise AttributeError(name)
