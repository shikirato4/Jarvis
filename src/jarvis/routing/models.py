from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.actions.models import ActionExecutionReceipt
from jarvis.cognition.models import OrchestrationResponse
from jarvis.core.metacommands import MetaCommand
from jarvis.core.models import RuntimeSnapshot
from jarvis.models.base import JarvisBaseModel
from jarvis.tools.models import ToolInvocationReceipt


class RouteType(StrEnum):
    METACOMMAND = "metacommand"
    ACTION = "action"
    TOOL = "tool"
    ORCHESTRATION = "orchestration"
    STATE = "state"


class TaskRequest(JarvisBaseModel):
    raw_input: str | None = None
    intent: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class TaskResponse(JarvisBaseModel):
    task_id: str
    route_type: RouteType
    target: str
    mode: str
    message: str
    meta_command: MetaCommand | None = None
    action_receipt: ActionExecutionReceipt | None = None
    tool_receipt: ToolInvocationReceipt | None = None
    orchestration: OrchestrationResponse | None = None
    state_snapshot: RuntimeSnapshot | None = None
