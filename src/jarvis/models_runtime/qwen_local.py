from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.config import Settings

from .base import ProviderKind
from .hybrid import ProviderCapabilities, ProviderOperationalState
from .local_profiles import LocalModelProfile, resolve_local_model_profile
from .openai_compatible import (
    ModelMismatchError,
    OpenAICompatibleError,
    OpenAICompatibleProvider,
)
from .qwen_identity import QwenIdentity

if TYPE_CHECKING:
    from .local_model_runtime import LocalModelRuntimeManager, LocalRuntimeStatus


class QwenLocalProviderUnavailableError(OpenAICompatibleError):
    pass


class QwenLocalModelMismatchError(ModelMismatchError):
    pass


class QwenLocalProvider(OpenAICompatibleProvider):
    provider_name = "qwen_local"
    provider_kind = ProviderKind.LOCAL

    def __init__(
        self,
        settings: Settings,
        *,
        profile: LocalModelProfile | None = None,
        client: httpx.Client | None = None,
        runtime_manager: LocalModelRuntimeManager | None = None,
    ) -> None:
        self.profile = profile or resolve_local_model_profile(settings)
        self.identity = QwenIdentity(
            self.profile.model_alias,
            self.profile.model_aliases,
            self.profile.expected_model_family,
        )
        super().__init__(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            base_url=settings.local_llm_base_url,
            configured_model=self.profile.model_alias,
            timeout_seconds=settings.local_llm_timeout_seconds,
            healthcheck_timeout_seconds=settings.local_llm_healthcheck_timeout_seconds,
            capabilities=ProviderCapabilities(
                text=True,
                images=self.profile.vision_capable,
                audio=self.profile.audio_capable,
                video=self.profile.video_capable,
                streaming=True,
                reasoning=True,
                structured_output=True,
                tool_calling=True,
                max_context_tokens=self.profile.context_size,
                input_formats=self._profile_input_formats(),
                native_transport_modalities=self.profile.native_transport_modalities,
                verification_source=(
                    f"configured {self.profile.expected_model_family} profile health required"
                ),
            ),
            client=client,
        )
        self._runtime_manager = runtime_manager
        self._runtime_client_is_injected = client is not None
        # A request without an explicit output ceiling lets llama.cpp consume
        # only whatever happens to remain in the 4K context after tool history.
        # Reserve the same bounded default used by desktop chat.
        self._default_output_tokens = min(
            max(settings.chat_default_max_tokens, 768),
            max(self.profile.context_size // 4, 128),
        )

    @property
    def runtime_manager(self) -> LocalModelRuntimeManager | None:
        return self._runtime_manager

    def ensure_runtime_ready(self) -> LocalRuntimeStatus | None:
        if self._runtime_manager is None:
            return None
        status = self._runtime_manager.ensure_ready()
        if status.endpoint and status.endpoint.rstrip("/") != self._base_url:
            self._base_url = status.endpoint.rstrip("/")
            if not self._runtime_client_is_injected and self._client is not None:
                self._client.close()
                self._client = None
        return status

    def prewarm(self) -> None:
        if self._runtime_manager is not None:
            self._runtime_manager.prewarm()

    def _healthy_state(self) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_ACTIVE

    def _request_payload(self, request, *, temperature, stream):
        payload = super()._request_payload(
            request, temperature=temperature, stream=stream
        )
        if "max_tokens" not in payload:
            payload["max_tokens"] = self._default_output_tokens
        if request.metadata.get("generation_profile") in {"detailed", "extended"}:
            budget_instruction = (
                "Termina la respuesta dentro del presupuesto de salida disponible. "
                "Prioriza las dimensiones solicitadas, evita repeticiones y no añadas "
                "una tabla o resumen final que duplique el contenido."
            )
            messages = payload.get("messages", [])
            system_message = next(
                (
                    message
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "system"
                ),
                None,
            )
            if system_message is None:
                messages.insert(
                    0, {"role": "system", "content": budget_instruction}
                )
            else:
                system_message["content"] = (
                    f"{str(system_message.get('content') or '').rstrip()}\n"
                    f"{budget_instruction}"
                )
        # The controller normally supplies an already-selected subset. Keep a
        # defensive provider boundary for direct callers: remove duplicate
        # schemas and do not expose terminal execution for an informational
        # "how/which command" question.
        tools_by_name: dict[str, dict[str, Any]] = {}
        for tool in payload.get("tools", []):
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and name:
                tools_by_name.setdefault(name, tool)
        payload["tools"] = list(tools_by_name.values())
        user_text = " ".join(
            str(message.get("content") or "")
            for message in payload.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "user"
        )
        folded = "".join(
            character
            for character in unicodedata.normalize("NFKD", user_text.casefold())
            if not unicodedata.combining(character)
        ).lstrip("¿¡ ")
        informational_terminal = bool(
            re.match(r"(?:como|que|how|what)\b", folded)
            and re.search(r"\b(?:comando|command|terminal|python|version)\b", folded)
        )
        if informational_terminal:
            payload["tools"] = [
                tool
                for tool in payload["tools"]
                if not str(tool.get("function", {}).get("name", "")).startswith("terminal.")
            ]
        # Jarvis keeps dotted action ids internally, while llama.cpp validates
        # function names against the OpenAI wire grammar. Encode only the wire
        # payload and restore canonical ids at the provider boundary.
        self._wire_tool_names = {
            name: name.replace(".", "__")
            for schema in request.tools
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and isinstance((name := schema["function"].get("name")), str)
            and "." in name
        }
        if self._wire_tool_names:
            for tool in payload.get("tools", []):
                function = tool.get("function") if isinstance(tool, dict) else None
                if isinstance(function, dict):
                    function["name"] = self._wire_tool_names.get(
                        function.get("name"), function.get("name")
                    )
            for message in payload.get("messages", []):
                for call in (
                    message.get("tool_calls", []) if isinstance(message, dict) else []
                ):
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict):
                        function["name"] = self._wire_tool_names.get(
                            function.get("name"), function.get("name")
                        )
            choice = payload.get("tool_choice")
            if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
                function = choice["function"]
                function["name"] = self._wire_tool_names.get(
                    function.get("name"), function.get("name")
                )
        # Tool relevance is owned by AgenticPromptPolicy. The provider only
        # applies the defensive direct-call rules above and wire-renames ids.
        # llama.cpp/Qwen exposes a verified binary switch for thinking. LOW is
        # served by that non-thinking mode so a modest completion budget cannot
        # be consumed entirely by hidden reasoning; MEDIUM/XHIGH keep thinking
        # enabled because the backend exposes no reliable intensity levels.
        preference = str(
            request.metadata.get("reasoning_preference") or "auto"
        ).casefold()
        if (
            self.profile.reasoning_adapter
            in {"qwen_enable_thinking", "qwen_enable_thinking_binary"}
            and preference in {"off", "low"}
            and preference in self.profile.reasoning_modes
        ):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif isinstance(request.metadata.get("chat_template_kwargs"), dict):
            payload["chat_template_kwargs"] = dict(
                request.metadata["chat_template_kwargs"]
            )
        return payload

    def _restore_tool_names(self, response):
        reverse = {
            wire: canonical
            for canonical, wire in getattr(self, "_wire_tool_names", {}).items()
        }
        if reverse:
            response.tool_calls = [
                call.model_copy(update={"name": reverse.get(call.name, call.name)})
                for call in response.tool_calls
            ]
        return response

    def infer(self, request, *, model_name, temperature, timeout_seconds):
        self.ensure_runtime_ready()
        response = self._restore_tool_names(
            super().infer(
                request,
                model_name=model_name,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        )
        response.metadata.update(self._profile_metadata())
        return response

    def stream_infer(
        self,
        request,
        *,
        model_name,
        temperature,
        timeout_seconds,
        cancel_check=None,
    ):
        self.ensure_runtime_ready()
        for chunk in super().stream_infer(
            request,
            model_name=model_name,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        ):
            chunk.metadata.update(self._profile_metadata())
            reverse = {
                wire: canonical
                for canonical, wire in getattr(self, "_wire_tool_names", {}).items()
            }
            if reverse and chunk.tool_calls:
                chunk.tool_calls = [
                    call.model_copy(update={"name": reverse.get(call.name, call.name)})
                    for call in chunk.tool_calls
                ]
            yield chunk

    def _healthy_message(self) -> str:
        return f"Qwen local disponible en {self.base_url}."

    def _validate_health_payload(self, payload: Any) -> None:
        super()._validate_health_payload(payload)
        advertised = payload.get("data")
        if not isinstance(advertised, list):
            raise TypeError("models response must contain a data list")
        model_ids = [
            item.get("id")
            for item in advertised
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        matched = self.identity.select_advertised(model_ids)
        if matched is None:
            raise QwenLocalModelMismatchError(
                f"El endpoint local no anuncia Qwen compatible con {self.identity.canonical_model}.",
                provider_name=self.provider_name,
            )
        self._configured_model = matched

    def _http_error(self, response: httpx.Response) -> OpenAICompatibleError:
        return QwenLocalProviderUnavailableError(
            self._start_diagnostic(f"HTTP {response.status_code}"),
            provider_name=self.provider_name,
            status_code=response.status_code,
        )

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        return QwenLocalProviderUnavailableError(
            self._start_diagnostic("sin conexión"), provider_name=self.provider_name
        )

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        return QwenLocalProviderUnavailableError(
            self._start_diagnostic("respuesta incompatible"),
            provider_name=self.provider_name,
        )

    def _model_mismatch_error(self, requested_model: str) -> ModelMismatchError:
        return QwenLocalModelMismatchError(
            f"LOCAL está fijado a {self.configured_model}; no se cambiará silenciosamente a {requested_model}.",
            provider_name=self.provider_name,
        )

    def _state_for_error(
        self, error: OpenAICompatibleError
    ) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_UNAVAILABLE

    def _start_diagnostic(self, reason: str) -> str:
        return f"El endpoint local Qwen {self.base_url} no está disponible ({reason}). Configura un servidor que anuncie {self.profile.expected_model_family} o un alias permitido."

    def _profile_metadata(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.id,
            "display_name": self.profile.display_name,
            "model_alias": self.profile.model_alias,
            "expected_model_family": self.profile.expected_model_family,
            "context_size": self.profile.context_size,
            "parallel": self.profile.parallel,
        }

    def _profile_input_formats(self) -> tuple[str, ...]:
        formats = ["text/plain"]
        if self.profile.vision_capable:
            formats.extend(("image/jpeg", "image/png"))
        if self.profile.audio_capable:
            formats.extend(("audio/wav", "audio/mpeg"))
        if self.profile.video_capable:
            formats.append("video/mp4")
        return tuple(formats)
