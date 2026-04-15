from __future__ import annotations

from jarvis.research_runtime.models import ResearchRunRequest
from jarvis.writing_runtime.models import WritingContinuationRequest

from .views import HudActionReceipt


class HudActionService:
    def __init__(self, *, ops_runtime, research_runtime, writing_runtime, indexing_runtime, unity_runtime, system_runtime) -> None:
        self._ops = ops_runtime
        self._research = research_runtime
        self._writing = writing_runtime
        self._indexing = indexing_runtime
        self._unity = unity_runtime
        self._system = system_runtime

    def recover_service(self, service_name: str, *, dry_run: bool = False) -> HudActionReceipt:
        receipt = self._ops.recover_service(service_name, dry_run=dry_run)
        return HudActionReceipt(action_name="recover_service", success=receipt.success, message=receipt.message, data=receipt.model_dump(mode="json"))

    def reset_breaker(self, service_name: str, dependency_name: str | None = None) -> HudActionReceipt:
        data = self._ops.reset_breaker(service_name, dependency_name)
        return HudActionReceipt(action_name="reset_breaker", success=bool(data.get("reset")), message="breaker reset processed", data=data)

    def retention_sweep(self) -> HudActionReceipt:
        receipt = self._ops.retention_sweep()
        return HudActionReceipt(action_name="retention_sweep", success=True, message="retention sweep completed", data=receipt.model_dump(mode="json"))

    def run_research(self, query: str, *, collection_name: str | None = None) -> HudActionReceipt:
        task = self._research.run(ResearchRunRequest(query=query, collection_name=collection_name))
        return HudActionReceipt(action_name="research_run", success=True, message=f"research task {task.status.value}", data=task.model_dump(mode="json"))

    def run_writing(self, prompt: str, *, target_window: str | None = None, collection_name: str | None = None) -> HudActionReceipt:
        receipt = self._writing.continue_writing(
            WritingContinuationRequest(prompt=prompt, target_window=target_window, collection_name=collection_name, write_directly=False)
        )
        return HudActionReceipt(action_name="writing_run", success=receipt.success, message=receipt.message, data=receipt.model_dump(mode="json"))

    def run_indexing(self) -> HudActionReceipt:
        receipt = self._indexing.run({"trigger": "manual", "requested_by": "hud"})
        return HudActionReceipt(action_name="indexing_run", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))

    def unity_bridge_status(self, project: str | None = None) -> HudActionReceipt:
        receipt = self._unity.bridge_health(project)
        return HudActionReceipt(action_name="unity_bridge_status", success=True, message="unity bridge status ready", data=receipt.model_dump(mode="json"))

    def system_status(self) -> HudActionReceipt:
        return HudActionReceipt(action_name="system_status", success=True, message="system runtime status ready", data=self._system.status())
