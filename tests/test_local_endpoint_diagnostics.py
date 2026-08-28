from __future__ import annotations

import httpx

from jarvis.environment import detect_local_endpoint
from jarvis.persistent_config import LOCAL_MODEL


def test_local_endpoint_timeout_is_typed_and_actionable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = httpx.Client(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )

    status = detect_local_endpoint(client=client, model=LOCAL_MODEL)

    assert status.status == "timeout"
    assert status.available is False
    assert status.suggestions


def test_local_endpoint_does_not_accept_a_different_model() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "different-model"}]})

    client = httpx.Client(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )

    status = detect_local_endpoint(client=client, model=LOCAL_MODEL)

    assert status.reachable is True
    assert status.model_found is False
    assert status.status == "model_missing"
