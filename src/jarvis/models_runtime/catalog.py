from __future__ import annotations

from typing import Iterable

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ModelProfile(JarvisBaseModel):
    logical_name: str
    provider: str
    provider_kind: str = "local"
    model_name: str
    purpose: str
    capabilities: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    temperature: float = 0.2
    timeout_seconds: float = 90.0
    priority: int = 100
    fallbacks: tuple[str, ...] = ()


class ModelCatalog:
    def __init__(self, profiles: Iterable[ModelProfile]) -> None:
        self._profiles = {profile.logical_name: profile for profile in profiles}

    def get(self, logical_name: str) -> ModelProfile | None:
        return self._profiles.get(logical_name)

    def list_profiles(self) -> list[ModelProfile]:
        return sorted(self._profiles.values(), key=lambda item: (item.priority, item.logical_name))

    def select_for_task(self, *, logical_name: str | None, task_type: str, required_capabilities: tuple[str, ...]) -> list[ModelProfile]:
        if logical_name:
            primary = self.get(logical_name)
            if primary is None:
                return []
            ordered = [primary]
            ordered.extend(profile for profile in self._fallback_profiles(primary.fallbacks) if profile.logical_name != primary.logical_name)
            return ordered

        matches = [
            profile
            for profile in self.list_profiles()
            if (not profile.task_types or task_type in profile.task_types)
            and all(capability in profile.capabilities for capability in required_capabilities)
        ]
        if matches:
            return matches
        return self.list_profiles()

    def _fallback_profiles(self, fallback_names: tuple[str, ...]) -> list[ModelProfile]:
        return [self._profiles[name] for name in fallback_names if name in self._profiles]


def build_default_model_catalog(settings=None) -> ModelCatalog:
    provider_name = getattr(settings, "model_provider_default", "ollama") if settings is not None else "ollama"
    general_provider = getattr(settings, "general_chat_model_provider", "gpt_oss") if settings is not None else "gpt_oss"
    provider_kind = "remote" if provider_name == "gpt_oss" else "local"
    general_provider_kind = "remote" if general_provider == "gpt_oss" else "local"
    provider_timeout = (
        getattr(settings, "gpt_oss_timeout_seconds", 90.0)
        if provider_name == "gpt_oss"
        else getattr(settings, "ollama_timeout_seconds", 90.0)
    )
    general_provider_timeout = (
        getattr(settings, "gpt_oss_timeout_seconds", 90.0)
        if general_provider == "gpt_oss"
        else getattr(settings, "ollama_timeout_seconds", 90.0)
    )
    general_model = getattr(settings, "gpt_oss_general_model", "gpt-oss-20b") if general_provider == "gpt_oss" else "llama3.1:8b"
    reasoning_model = getattr(settings, "gpt_oss_reasoning_model", "qwen2.5:14b") if provider_name == "gpt_oss" else "qwen2.5:14b"
    coding_model = getattr(settings, "gpt_oss_coding_model", "qwen2.5-coder:14b") if provider_name == "gpt_oss" else "qwen2.5-coder:14b"
    summarizer_model = getattr(settings, "gpt_oss_summarizer_model", "llama3.1:8b") if provider_name == "gpt_oss" else "llama3.1:8b"
    writing_model = getattr(settings, "gpt_oss_writing_model", "mistral-nemo:12b") if provider_name == "gpt_oss" else "mistral-nemo:12b"
    planner_model = getattr(settings, "gpt_oss_planner_model", "qwen2.5:14b") if provider_name == "gpt_oss" else "qwen2.5:14b"
    profiles = [
        ModelProfile(
            logical_name="general_assistant",
            provider=general_provider,
            provider_kind=general_provider_kind,
            model_name=general_model,
            purpose="General assistant dialogue and classification.",
            capabilities=("chat", "classification"),
            task_types=("assistant", "classification"),
            temperature=0.2,
            timeout_seconds=general_provider_timeout,
            priority=10,
            fallbacks=tuple(getattr(settings, "general_chat_model_fallback_order", ()) or ()),
        ),
        ModelProfile(
            logical_name="reasoning_engine",
            provider=provider_name,
            provider_kind=provider_kind,
            model_name=reasoning_model,
            purpose="Reasoning and plan generation.",
            capabilities=("chat", "planning", "reasoning"),
            task_types=("reasoning", "planning"),
            temperature=0.1,
            timeout_seconds=provider_timeout,
            priority=5,
            fallbacks=("general_assistant",),
        ),
        ModelProfile(
            logical_name="coding_engine",
            provider=provider_name,
            provider_kind=provider_kind,
            model_name=coding_model,
            purpose="Coding, implementation and technical transformation tasks.",
            capabilities=("chat", "coding", "reasoning"),
            task_types=("coding",),
            temperature=0.1,
            timeout_seconds=provider_timeout,
            priority=15,
            fallbacks=("reasoning_engine", "general_assistant"),
        ),
        ModelProfile(
            logical_name="summarizer",
            provider=provider_name,
            provider_kind=provider_kind,
            model_name=summarizer_model,
            purpose="Summaries and brief synthesis.",
            capabilities=("chat", "summarization"),
            task_types=("summarization",),
            temperature=0.1,
            timeout_seconds=provider_timeout,
            priority=20,
            fallbacks=("general_assistant",),
        ),
        ModelProfile(
            logical_name="writing_engine",
            provider=provider_name,
            provider_kind=provider_kind,
            model_name=writing_model,
            purpose="Long-form structured writing.",
            capabilities=("chat", "writing"),
            task_types=("writing",),
            temperature=0.35,
            timeout_seconds=provider_timeout,
            priority=12,
            fallbacks=("general_assistant",),
        ),
        ModelProfile(
            logical_name="planner",
            provider=provider_name,
            provider_kind=provider_kind,
            model_name=planner_model,
            purpose="Task and action planning.",
            capabilities=("chat", "planning", "classification"),
            task_types=("planning", "classification"),
            temperature=0.1,
            timeout_seconds=provider_timeout,
            priority=8,
            fallbacks=("reasoning_engine", "general_assistant"),
        ),
    ]
    return ModelCatalog(profiles)
