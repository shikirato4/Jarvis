from __future__ import annotations

import httpx

from jarvis.config import Settings
from jarvis.models_runtime.qwen_identity import local_model_policy_description
from jarvis.models_runtime.qwen_local import QwenLocalProvider
from jarvis.models_runtime.base import ModelMessage, ModelRequest


class _RuntimeStatus:
    endpoint = "http://127.0.0.1:8080/v1"


class _RuntimeManager:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.prewarm_calls = 0

    def ensure_ready(self):
        self.ensure_calls += 1
        return _RuntimeStatus()

    def prewarm(self) -> None:
        self.prewarm_calls += 1


def _settings(**overrides):
    return Settings(
        local_runtime_profile_id="qwen38-27b-iq4-xs",
        local_llm_model="Qwen3.8-quantized",
        local_llm_model_aliases=("Qwen/Qwen3-8B-GGUF",),
        local_llm_base_url="http://qwen.test/v1",
        **overrides,
    )


def test_qwen_provider_accepts_configured_alias_and_rejects_other_family() -> None:
    provider = QwenLocalProvider(_settings())
    assert provider.provider_name == "qwen_local"
    assert provider.identity.is_compatible("Qwen/Qwen3-8B-GGUF")
    assert provider.identity.select_advertised(["other-model"]) is None


def test_qwen_provider_health_matches_advertised_alias() -> None:
    provider = QwenLocalProvider(_settings())
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"data": [{"id": "Qwen/Qwen3-8B-GGUF"}]},
            request=request,
        )
    )
    provider._client = httpx.Client(transport=transport, base_url=provider.base_url)  # noqa: SLF001
    status = provider.status_check()
    assert status.healthy is True
    assert provider.configured_model == "Qwen/Qwen3-8B-GGUF"


def test_qwen_provider_ensures_runtime_before_real_inference() -> None:
    runtime = _RuntimeManager()
    provider = QwenLocalProvider(_settings(), runtime_manager=runtime)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            json={
                "model": provider.configured_model,
                "choices": [{"message": {"content": "local ready"}}],
            },
        )
    )
    provider._client = httpx.Client(  # noqa: SLF001
        transport=transport, base_url="http://127.0.0.1:8080/v1/"
    )
    provider._runtime_client_is_injected = True  # noqa: SLF001

    response = provider.infer(
        ModelRequest(task_type="assistant", prompt="hola"),
        model_name=provider.configured_model,
        temperature=0.0,
        timeout_seconds=1.0,
    )

    assert runtime.ensure_calls == 1
    assert provider.base_url == "http://127.0.0.1:8080/v1"
    assert response.content == "local ready"


def test_qwen_sync_response_never_retains_raw_reasoning_content() -> None:
    provider = QwenLocalProvider(_settings())
    private = "PRIVATE-COT-MUST-NOT-PERSIST"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            json={
                "model": provider.configured_model,
                "choices": [
                    {
                        "message": {
                            "content": "respuesta pública",
                            "reasoning_content": private,
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    provider._client = httpx.Client(  # noqa: SLF001
        transport=transport, base_url=provider.base_url
    )

    response = provider.infer(
        ModelRequest(task_type="assistant", prompt="hola"),
        model_name=provider.configured_model,
        temperature=0.0,
        timeout_seconds=1.0,
    )

    assert response.content == "respuesta pública"
    assert private not in repr(response.raw)
    assert "reasoning_content" not in repr(response.model_dump(mode="json"))


def test_qwen_local_policy_never_describes_nemotron_or_nvfp4() -> None:
    description = local_model_policy_description()

    assert "Qwen3.5-9B" in description
    assert "quantized" in description
    assert "quantized" in description
    assert "Nemotron" not in description
    assert "NVFP4" not in description


def test_qwen_local_maps_only_verified_reasoning_off_switch() -> None:
    provider = QwenLocalProvider(_settings())
    request = ModelRequest(
        task_type="assistant",
        messages=[ModelMessage(role="user", content="hola")],
        metadata={"reasoning_preference": "off"},
    )
    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    other = request.model_copy(update={"metadata": {"reasoning_preference": "xhigh"}})
    other_payload = provider._request_payload(other, temperature=0.0, stream=False)  # noqa: SLF001
    assert "chat_template_kwargs" not in other_payload

    low = request.model_copy(update={"metadata": {"reasoning_preference": "low"}})
    low_payload = provider._request_payload(low, temperature=0.0, stream=False)  # noqa: SLF001
    assert low_payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen_local_disables_thinking_when_completion_budget_is_small() -> None:
    """Cupo corto + thinking = respuesta vacia: se apaga solo el thinking."""

    provider = QwenLocalProvider(_settings())
    request = ModelRequest(
        task_type="assistant",
        messages=[ModelMessage(role="user", content="que hora es")],
        max_tokens=160,
        metadata={"reasoning_preference": "medium"},
    )

    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    # Con cupo amplio el razonamiento sigue disponible.
    amplio = request.model_copy(update={"max_tokens": 4096})
    amplio_payload = provider._request_payload(  # noqa: SLF001
        amplio, temperature=0.0, stream=False
    )
    assert "chat_template_kwargs" not in amplio_payload


def test_qwen_detailed_profile_is_instructed_to_finish_without_repetition() -> None:
    provider = QwenLocalProvider(_settings())
    request = ModelRequest(
        task_type="assistant",
        messages=[
            ModelMessage(role="system", content="Eres Jarvis."),
            ModelMessage(role="user", content="Compara dos bandas en detalle."),
        ],
        max_tokens=1_200,
        metadata={"generation_profile": "detailed"},
    )

    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001

    system = payload["messages"][0]["content"]
    assert system.startswith("Eres Jarvis.")
    assert "dentro del presupuesto de salida" in system
    assert "evita repeticiones" in system
    assert payload["max_tokens"] == 1_200
    assert request.messages[0].content == "Eres Jarvis."


def test_qwen_local_encodes_dotted_action_names_for_llama_wire_protocol() -> None:
    provider = QwenLocalProvider(
        Settings(
            _env_file=None,
            local_llm_model_aliases=("Qwen3.8-27B-IQ4_XS",),
        )
    )
    request = ModelRequest(
        task_type="agent",
        messages=[
            ModelMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "calculator.evaluate",
                        "arguments": {"expression": "1 + 1"},
                    }
                ],
            )
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator.evaluate",
                    "description": "safe calculator",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    assert payload["tools"][0]["function"]["name"] == "calculator__evaluate"
    assert (
        payload["messages"][0]["tool_calls"][0]["function"]["name"]
        == "calculator__evaluate"
    )


def test_qwen_local_preserves_agent_selected_large_tool_registry() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"unrelated_{index}",
                "description": "unrelated capability with a verbose schema " * 20,
                "parameters": {"type": "object"},
            },
        }
        for index in range(72)
    ]
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "calculator.evaluate",
                "description": "Evaluate arithmetic expressions",
                "parameters": {"type": "object"},
            },
        }
    )
    request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="Calcula 781 * 94")],
        tools=tools,
    )
    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert len(names) == len(tools)
    assert "calculator__evaluate" in names


def test_qwen_local_preserves_agent_selected_tool_order() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"generic.action_{index}",
                "description": "Use a project session and return information",
                "parameters": {"type": "object"},
            },
        }
        for index in range(30)
    ]
    tools.extend(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Persistent PowerShell terminal operation",
                "parameters": {"type": "object"},
            },
        }
        for name in ("terminal.start_session", "terminal.run")
    )
    request = ModelRequest(
        task_type="agent",
        messages=[
            ModelMessage(
                role="system",
                content="Available namespaces include files, terminal and operations.",
            ),
            ModelMessage(
                role="user",
                content="Use terminal.start_session and terminal.run in PowerShell",
            ),
        ],
        tools=tools,
    )

    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert names[:2] == ["generic__action_0", "generic__action_1"]
    assert names[-2:] == ["terminal__start_session", "terminal__run"]


def test_qwen_local_keeps_file_search_for_clear_file_intent() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"generic.action_{index}",
                "description": "Unrelated project capability",
                "parameters": {"type": "object"},
            },
        }
        for index in range(30)
    ]
    tools.extend(
        [
            {
                "type": "function",
                "function": {
                    "name": "files.search",
                    "description": "Search/buscar a file or archivo by filename",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "files.stat",
                    "description": "Inspect file size and metadata",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal.run",
                    "description": "Run a terminal command",
                    "parameters": {"type": "object"},
                },
            },
        ]
    )
    request = ModelRequest(
        task_type="agent",
        messages=[
            ModelMessage(role="user", content="Busca un archivo llamado report.pdf")
        ],
        tools=tools,
    )
    payload = provider._request_payload(request, temperature=0.0, stream=False)  # noqa: SLF001
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert "files__search" in names
    assert "terminal__run" in names


def test_qwen_local_does_not_reclassify_action_versus_information_tools() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"generic.action_{index}",
                "description": "Unrelated capability",
                "parameters": {"type": "object"},
            },
        }
        for index in range(20)
    ]
    tools.extend(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Persistent PowerShell terminal operation",
                "parameters": {"type": "object"},
            },
        }
        for name in ("terminal.start_session", "terminal.run")
    )
    action = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="limpia mi cache DNS")],
        tools=tools,
    )
    informational = action.model_copy(
        update={
            "messages": [
                ModelMessage(role="user", content="¿Cómo puedo limpiar mi cache DNS?")
            ]
        }
    )

    action_names = [
        item["function"]["name"]
        for item in provider._request_payload(  # noqa: SLF001
            action, temperature=0.0, stream=False
        )["tools"]
    ]
    informational_names = [
        item["function"]["name"]
        for item in provider._request_payload(  # noqa: SLF001
            informational, temperature=0.0, stream=False
        )["tools"]
    ]

    assert {"terminal__start_session", "terminal__run"} <= set(action_names)
    assert action_names == informational_names


def test_qwen_provider_does_not_expand_schemas_after_observation() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"generic.action_{index}",
                "description": "Unrelated capability",
                "parameters": {"type": "object"},
            },
        }
        for index in range(20)
    ]
    tools.extend(
        [
            {
                "type": "function",
                "function": {
                    "name": "files.search",
                    "description": "Search files by name in a bounded scope",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal.start_session",
                    "description": "Start PowerShell",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal.run",
                    "description": "Run PowerShell",
                    "parameters": {"type": "object"},
                },
            },
        ]
    )
    user = ModelMessage(
        role="user",
        content="busca en mis archivos la ruta de Hogwarts Legacy",
    )
    first = ModelRequest(task_type="agent", messages=[user], tools=tools)
    continued = first.model_copy(
        update={
            "messages": [
                user,
                ModelMessage(
                    role="tool",
                    name="files.search",
                    tool_call_id="search-1",
                    content='{"status":"success","data":{"items":[]}}',
                ),
            ]
        }
    )

    first_names = [
        item["function"]["name"]
        for item in provider._request_payload(  # noqa: SLF001
            first, temperature=0.0, stream=False
        )["tools"]
    ]
    continued_names = [
        item["function"]["name"]
        for item in provider._request_payload(  # noqa: SLF001
            continued, temperature=0.0, stream=False
        )["tools"]
    ]

    assert "files__search" in first_names
    assert {"terminal__start_session", "terminal__run"} <= set(first_names)
    assert first_names == continued_names


def test_qwen_local_reserves_output_and_preserves_selected_file_subset() -> None:
    provider = QwenLocalProvider(Settings(_env_file=None))
    requested_names = {
        "files.create_directory",
        "files.create_file",
        "files.stat",
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Perform {name} for a file request",
                "parameters": {"type": "object"},
            },
        }
        for name in [
            "files.resolve_path",
            "files.list",
            "files.search",
            "files.stat",
            "files.read_text",
            "files.create_directory",
            "files.create_file",
            "files.copy",
            "files.move",
            "files.rename",
            "files.delete",
            "files.batch_move",
            "files.open",
            "files.reveal",
        ]
    ]
    tools.extend(
        {
            "type": "function",
            "function": {
                "name": f"generic.action_{index}",
                "description": "Unrelated capability",
                "parameters": {"type": "object"},
            },
        }
        for index in range(20)
    )
    request = ModelRequest(
        task_type="agent",
        messages=[
            ModelMessage(
                role="user",
                content=(
                    "Usa files.create_directory, files.create_file y files.stat "
                    "para crear un archivo y confirmar su tamaño."
                ),
            )
        ],
        tools=tools,
    )
    original_names = [tool["function"]["name"] for tool in request.tools]

    payload = provider._request_payload(  # noqa: SLF001
        request, temperature=0.0, stream=False
    )
    wire_names = [tool["function"]["name"] for tool in payload["tools"]]

    assert payload["max_tokens"] == 768
    assert len(wire_names) == len(tools)
    assert {name.replace(".", "__") for name in requested_names} <= set(wire_names)
    assert [tool["function"]["name"] for tool in request.tools] == original_names


def test_qwen_local_preserves_explicit_agent_benchmark_cap() -> None:
    provider = QwenLocalProvider(_settings())
    request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="benchmark")],
        max_tokens=260,
        metadata={"disable_auto_continuation": True},
    )

    payload = provider._request_payload(  # noqa: SLF001
        request, temperature=0.0, stream=False
    )

    assert payload["max_tokens"] == 260
