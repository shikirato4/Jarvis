from __future__ import annotations

import base64
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
            message = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tool_calls = _normalize_tool_calls(message.get("tool_calls"))
            usage_payload = body.get("usage") or {}
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
            raw=body,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            usage=ModelUsage(
                input_tokens=usage_payload.get("prompt_tokens"),
                output_tokens=usage_payload.get("completion_tokens"),
                total_tokens=usage_payload.get("total_tokens"),
            ),
            metadata={"task_type": request.task_type},
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
                        yield StreamChunk(done=True, metadata={"chunks": chunks})
                        return
                    event = json.loads(data)
                    delta = ((event.get("choices") or [{}])[0] or {}).get("delta") or {}
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
        yield StreamChunk(
            done=True, metadata={"chunks": chunks, "reason": "stream_closed"}
        )

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
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = request.response_format
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
