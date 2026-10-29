from __future__ import annotations

import base64
import copy
import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from .base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ProviderHealth,
    ProviderKind,
    StreamChunk,
)
from .hybrid import ProviderCapabilities, ProviderOperationalState, ProviderStatus


class OpenAICompatibleError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class ModelMismatchError(OpenAICompatibleError):
    pass


class UnsupportedMediaTransportError(OpenAICompatibleError):
    pass


class OpenAICompatibleProvider:
    provider_name: str
    provider_kind: ProviderKind

    def __init__(
        self,
        *,
        provider_name: str,
        provider_kind: ProviderKind,
        base_url: str,
        configured_model: str,
        timeout_seconds: float,
        healthcheck_timeout_seconds: float,
        capabilities: ProviderCapabilities,
        client: httpx.Client | None = None,
        headers_factory: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.provider_kind = provider_kind
        self._base_url = base_url.rstrip("/")
        self._configured_model = configured_model
        self._timeout_seconds = timeout_seconds
        self._healthcheck_timeout_seconds = healthcheck_timeout_seconds
        self._headers_factory = headers_factory or (lambda: {})
        # SSL context creation can be surprisingly expensive on Windows. Keep
        # provider construction cheap and initialize transport only when a
        # request is actually made. Injected clients remain eager by design.
        self._client = client
        self.capabilities = capabilities

    @property
    def configured_model(self) -> str:
        return self._configured_model

    @property
    def base_url(self) -> str:
        return self._base_url

    def status_check(self) -> ProviderStatus:
        started_at = time.perf_counter()
        try:
            response = self._http_client().get(
                "models",
                headers=self._request_headers(),
                timeout=self._healthcheck_timeout_seconds,
            )
            self._raise_for_status(response)
            payload = response.json()
            self._validate_health_payload(payload)
        except OpenAICompatibleError as exc:
            return self._status_from_error(exc, started_at)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            error = self._unavailable_error(exc)
            return self._status_from_error(error, started_at)
        except (ValueError, TypeError, KeyError) as exc:
            error = self._protocol_error(exc)
            return self._status_from_error(error, started_at)
        return ProviderStatus(
            provider_name=self.provider_name,
            state=self._healthy_state(),
            healthy=True,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            message=self._healthy_message(),
        )

    def health_check(self) -> ProviderHealth:
        status = self.status_check()
        return ProviderHealth(
            provider_name=self.provider_name,
            healthy=status.healthy,
            checked_at=status.checked_at,
            details={
                "state": status.state.value,
                "latency_ms": status.latency_ms,
                "message": status.message,
                "retry_after_seconds": status.retry_after_seconds,
            },
        )

    def infer(
        self,
        request: ModelRequest,
        *,
        model_name: str,
        temperature: float | None,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        self._validate_requested_model(model_name)
        payload = self._request_payload(request, temperature=temperature, stream=False)
        started_at = time.perf_counter()
        try:
            response = self._http_client().post(
                "chat/completions",
                json=payload,
                headers=self._request_headers(),
                timeout=timeout_seconds or self._timeout_seconds,
            )
            self._raise_for_status(response)
            body = response.json()
            choice = (body.get("choices") or [{}])[0] or {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tool_calls = _normalize_tool_calls(message.get("tool_calls"))
            usage_payload = body.get("usage") or {}
            finish_reason = choice.get("finish_reason")
        except OpenAICompatibleError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise self._unavailable_error(exc) from exc
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise self._protocol_error(exc) from exc
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or self._configured_model,
            model_name=str(body.get("model") or self._configured_model),
            content=content,
            tool_calls=tool_calls,
            raw=_without_private_reasoning(body),
            latency_ms=(time.perf_counter() - started_at) * 1000,
            usage=ModelUsage(
                input_tokens=usage_payload.get("prompt_tokens"),
                output_tokens=usage_payload.get("completion_tokens"),
                total_tokens=usage_payload.get("total_tokens"),
            ),
            metadata={
                "task_type": request.task_type,
                "finish_reason": (
                    str(finish_reason) if finish_reason is not None else None
                ),
                "stream_closed_normally": True,
                "incomplete": finish_reason is None
                or str(finish_reason) not in {"stop", "tool_calls"},
                "completion_contract_enforced": True,
                "provider_text_length": len(content),
                "prompt_tokens": usage_payload.get("prompt_tokens"),
                "completion_tokens": usage_payload.get("completion_tokens"),
                "requested_max_tokens": payload.get("max_tokens"),
            },
        )

    def stream_infer(
        self,
        request: ModelRequest,
        *,
        model_name: str,
        temperature: float | None,
        timeout_seconds: float | None,
        cancel_check=None,
    ) -> Iterator[StreamChunk]:
        self._validate_requested_model(model_name)
        payload = self._request_payload(request, temperature=temperature, stream=True)
        chunks = 0
        reasoning_chunks = 0
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        streamed_tool_calls: dict[int, dict[str, str]] = {}
        tool_calls_emitted = False
        try:
            with self._http_client().stream(
                "POST",
                "chat/completions",
                json=payload,
                headers=self._request_headers(),
                timeout=timeout_seconds or self._timeout_seconds,
            ) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if cancel_check is not None and cancel_check():
                        yield StreamChunk(
                            done=True, metadata={"cancelled": True, "chunks": chunks}
                        )
                        return
                    if not line:
                        continue
                    data = (
                        line[5:].strip() if line.startswith("data:") else line.strip()
                    )
                    if not data:
                        continue
                    if data == "[DONE]":
                        tool_calls = _normalize_streamed_tool_calls(streamed_tool_calls)
                        # A valid provider may send a finish_reason-only final
                        # choice (for example an empty but successful tool
                        # turn).  Do not misclassify that as a protocol error;
                        # only a stream with neither visible output nor a
                        # terminal reason is malformed.
                        if chunks == 0 and not tool_calls and finish_reason is None:
                            raise self._protocol_error(
                                ValueError("model stream contained no visible output")
                            )
                        yield StreamChunk(
                            done=True,
                            tool_calls=([] if tool_calls_emitted else tool_calls),
                            metadata={
                                "chunks": chunks,
                                "reasoning_chunks": reasoning_chunks,
                                "finish_reason": finish_reason,
                                "stream_closed_normally": True,
                                "incomplete": finish_reason
                                not in {"stop", "tool_calls"},
                                "completion_contract_enforced": True,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "requested_max_tokens": payload.get("max_tokens"),
                            },
                        )
                        return
                    event = json.loads(data)
                    usage = event.get("usage") or {}
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get(
                            "completion_tokens", completion_tokens
                        )
                    choice = (event.get("choices") or [{}])[0] or {}
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("reasoning_content"), str) and delta.get(
                        "reasoning_content"
                    ):
                        reasoning_chunks += 1
                        # Reasoning is exposed live to the local owner for
                        # display; persistence boundaries still strip it.
                        yield StreamChunk(
                            reasoning=str(delta.get("reasoning_content")),
                            metadata={
                                "provider": self.provider_name,
                                "model": self._configured_model,
                                "reasoning_active": True,
                            },
                        )
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice.get("finish_reason"))
                    for call_delta in delta.get("tool_calls") or []:
                        index = int(call_delta.get("index", 0))
                        accumulated = streamed_tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if call_delta.get("id"):
                            accumulated["id"] = str(call_delta["id"])
                        function = call_delta.get("function") or {}
                        if function.get("name"):
                            accumulated["name"] += str(function["name"])
                        if function.get("arguments"):
                            accumulated["arguments"] += str(function["arguments"])
                    text = delta.get("content") or ""
                    if text:
                        chunks += 1
                        yield StreamChunk(
                            text=str(text),
                            metadata={
                                "provider": self.provider_name,
                                "model": self._configured_model,
                            },
                        )
                    if choice.get("finish_reason") == "tool_calls":
                        tool_calls = _normalize_streamed_tool_calls(streamed_tool_calls)
                        if tool_calls:
                            tool_calls_emitted = True
                            yield StreamChunk(
                                tool_calls=tool_calls,
                                metadata={
                                    "provider": self.provider_name,
                                    "model": self._configured_model,
                                },
                            )
        except OpenAICompatibleError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise self._unavailable_error(exc) from exc
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            raise self._protocol_error(exc) from exc
        trailing_tool_calls = _normalize_streamed_tool_calls(streamed_tool_calls)
        if chunks == 0 and not trailing_tool_calls and finish_reason is None:
            raise self._protocol_error(
                ValueError("model stream closed without visible output")
            )
        yield StreamChunk(
            done=True,
            tool_calls=([] if tool_calls_emitted else trailing_tool_calls),
            metadata={
                "chunks": chunks,
                "reason": "stream_closed",
                "reasoning_chunks": reasoning_chunks,
                "finish_reason": finish_reason,
                "stream_closed_normally": False,
                "incomplete": True,
                "completion_contract_enforced": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "requested_max_tokens": payload.get("max_tokens"),
            },
        )

    def list_models(self) -> list[str]:
        """Return model identifiers advertised by this explicitly configured endpoint."""
        return [item["identifier"] for item in self.discover_models()]

    def discover_models(self) -> list[dict[str, Any]]:
        """Return safe identity metadata from the configured endpoint's /models API."""
        try:
            response = self._http_client().get(
                "models",
                headers=self._request_headers(),
                timeout=self._healthcheck_timeout_seconds,
            )
            self._raise_for_status(response)
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("models response must be an object")
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise TypeError("models.data must be a list")
            discovered: dict[str, dict[str, Any]] = {}
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                identifier = item["id"].strip()
                if not identifier:
                    continue
                version = identifier.rsplit(":", 1)[1] if ":" in identifier else None
                discovered[identifier] = {
                    "provider": self.provider_name,
                    "model": identifier,
                    "identifier": identifier,
                    "version": version,
                    "owned_by": item.get("owned_by") if isinstance(item.get("owned_by"), str) else None,
                }
            return [discovered[key] for key in sorted(discovered)]
        except OpenAICompatibleError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise self._unavailable_error(exc) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise self._protocol_error(exc) from exc

    def get_model_info(self) -> dict[str, Any]:
        """Expose configured identity and conservative, operator-verified capabilities."""
        return {
            "provider": self.provider_name,
            "model": self._configured_model,
            "identifier": self._configured_model,
            "version": self._configured_model.rsplit(":", 1)[1] if ":" in self._configured_model else None,
            "capabilities": self.capabilities.model_dump(mode="json"),
        }

    def supports(self, capability: str) -> bool:
        normalized = capability.strip().casefold().replace("-", "_")
        field = {
            "chat": "text",
            "vision": "images",
            "tool_calling": "tool_calling",
            "structured_output": "structured_output",
            "reasoning": "reasoning",
            "streaming": "streaming",
            "long_context": "max_context_tokens",
        }.get(normalized, normalized)
        value = getattr(self.capabilities, field, False)
        if field == "max_context_tokens":
            return value is not None
        return bool(value)

    def _http_client(self) -> httpx.Client:
        client = self._client
        if client is None:
            # Plain local endpoints do not need a certificate store. Passing
            # ``verify=False`` avoids httpx building the Windows CA context for
            # LM Studio/vLLM/llama.cpp URLs while HTTPS providers retain full
            # verification.
            verify = not self._base_url.casefold().startswith("http://")
            client = httpx.Client(
                base_url=f"{self._base_url}/",
                verify=verify,
                trust_env=verify,
            )
            self._client = client
        return client

    def _request_payload(
        self, request: ModelRequest, *, temperature: float | None, stream: bool
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            _provider_message(message) for message in request.messages
        ]
        if not messages and request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        if request.media:
            messages = self._attach_ephemeral_media(messages, request)
        payload: dict[str, Any] = {
            "model": self._configured_model,
            "messages": messages,
            "stream": stream,
        }
        resolved_temperature = (
            temperature if temperature is not None else request.temperature
        )
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            # Provider adapters rewrite function names for their wire format.
            # Keep the canonical request immutable across diagnostics/retries.
            payload["tools"] = copy.deepcopy(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = copy.deepcopy(request.tool_choice)
        if request.response_format is not None:
            payload["response_format"] = copy.deepcopy(request.response_format)
        if isinstance(request.metadata.get("chat_template_kwargs"), dict):
            payload["chat_template_kwargs"] = dict(
                request.metadata["chat_template_kwargs"]
            )
        return payload

    def _attach_ephemeral_media(
        self,
        messages: list[dict[str, Any]],
        request: ModelRequest,
    ) -> list[dict[str, Any]]:
        updated = [dict(message) for message in messages]
        user_index = next(
            (
                index
                for index in range(len(updated) - 1, -1, -1)
                if updated[index].get("role") == "user"
            ),
            None,
        )
        if user_index is None:
            updated.append({"role": "user", "content": request.prompt or ""})
            user_index = len(updated) - 1
        original_content = updated[user_index].get("content") or request.prompt or ""
        parts: list[dict[str, Any]] = []
        if isinstance(original_content, str) and original_content:
            parts.append({"type": "text", "text": original_content})
        for media in request.media:
            if (
                media.category.casefold() != "image"
                or media.mime_type.casefold() not in {"image/png", "image/jpeg"}
            ):
                raise UnsupportedMediaTransportError(
                    f"El transporte {media.category}/{media.mime_type} no está verificado para {self.provider_name}.",
                    provider_name=self.provider_name,
                )
            encoded = base64.b64encode(media.data).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media.mime_type};base64,{encoded}"},
                }
            )
        updated[user_index]["content"] = parts
        return updated

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **self._headers_factory()}

    def _validate_requested_model(self, model_name: str) -> None:
        if model_name != self._configured_model:
            raise self._model_mismatch_error(model_name)

    def _validate_health_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("models response must be an object")

    def _status_from_error(
        self, error: OpenAICompatibleError, started_at: float
    ) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.provider_name,
            state=self._state_for_error(error),
            healthy=False,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            message=str(error),
            retry_after_seconds=error.retry_after_seconds,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        raise self._http_error(response)

    def _healthy_state(self) -> ProviderOperationalState:
        raise NotImplementedError

    def _healthy_message(self) -> str:
        raise NotImplementedError

    def _http_error(self, response: httpx.Response) -> OpenAICompatibleError:
        raise NotImplementedError

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        raise NotImplementedError

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        raise NotImplementedError

    def _model_mismatch_error(self, requested_model: str) -> ModelMismatchError:
        raise NotImplementedError

    def _state_for_error(
        self, error: OpenAICompatibleError
    ) -> ProviderOperationalState:
        raise NotImplementedError


_PRIVATE_REASONING_KEYS = {
    "reasoning_content",
    "raw_reasoning_content",
    "chain_of_thought",
    "hidden_reasoning_tokens",
}


def _without_private_reasoning(value: Any) -> Any:
    """Remove private reasoning recursively before a provider payload escapes."""

    if isinstance(value, dict):
        return {
            key: _without_private_reasoning(item)
            for key, item in value.items()
            if str(key).casefold() not in _PRIVATE_REASONING_KEYS
        }
    if isinstance(value, list):
        return [_without_private_reasoning(item) for item in value]
    return value


def _normalize_tool_calls(payload: Any) -> list[ModelToolCall]:
    if not isinstance(payload, list):
        return []
    normalized: list[ModelToolCall] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        call_id = item.get("id")
        name = function.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        raw_arguments = function.get("arguments") or {}
        arguments: dict[str, Any] | str
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments
            else:
                arguments = parsed if isinstance(parsed, dict) else raw_arguments
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = str(raw_arguments)
        normalized.append(ModelToolCall(id=call_id, name=name, arguments=arguments))
    return normalized


def _normalize_streamed_tool_calls(
    accumulated: dict[int, dict[str, str]],
) -> list[ModelToolCall]:
    payload = [
        {
            "id": item.get("id") or f"stream-tool-{index}",
            "function": {
                "name": item.get("name", ""),
                "arguments": item.get("arguments", "{}"),
            },
        }
        for index, item in sorted(accumulated.items())
        if item.get("name")
    ]
    return _normalize_tool_calls(payload)


def _provider_message(message: ModelMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": (
                        call.arguments
                        if isinstance(call.arguments, str)
                        else json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return payload
