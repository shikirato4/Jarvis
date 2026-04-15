from __future__ import annotations

from datetime import datetime, timezone
import re
from uuid import uuid4

from .models import DesktopAgentMissionRequest, DesktopAgentPhase, DesktopWorldState


class DesktopWorldModelBuilder:
    def create(self, request: DesktopAgentMissionRequest | dict) -> DesktopWorldState:
        payload = DesktopAgentMissionRequest.model_validate(request)
        mission_id = str(uuid4())
        goal = payload.goal.strip()
        target_application = self._extract_application(goal)
        return DesktopWorldState(
            mission_id=mission_id,
            goal_id=f"goal-{uuid4().hex[:8]}",
            current_goal=goal,
            phase=DesktopAgentPhase.PENDING,
            target_application=target_application,
            metadata={
                "request": payload.model_dump(mode="json"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "initial_strategy": "grounded_desktop_agent",
            },
        )

    @staticmethod
    def _extract_application(goal: str) -> str | None:
        match = re.search(r"abre\s+(.+?)(?:\s+y|$)", goal, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip("\"' ")
        if "ventana activa" in goal.casefold():
            return None
        return None
