"""Voice runtime package with lazy service loading."""

__all__ = ["VoiceRuntimeService"]


def __getattr__(name: str):
    if name == "VoiceRuntimeService":
        from .service import VoiceRuntimeService

        return VoiceRuntimeService
    raise AttributeError(name)
