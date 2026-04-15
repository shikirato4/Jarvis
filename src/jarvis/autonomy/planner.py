from __future__ import annotations

import json
from uuid import uuid4

from jarvis.models_runtime.base import ModelMessage, ModelRequest

from .base import MissionPlan, MissionStep, MissionStepKind, MissionStepStatus, RiskLevel
from .strategies import classify_goal


class MissionPlanner:
    def __init__(self, models, settings, logger=None) -> None:
        self._models = models
        self._settings = settings
        self._logger = logger

    def plan(self, mission) -> MissionPlan:
        strategy_name, sequence = classify_goal(mission.goal.objective, mission.context.payload)
        model_steps = self._model_plan(mission)
        steps = model_steps or self._fallback_steps(mission, sequence)
        return MissionPlan(
            mission_id=mission.mission_id,
            summary=f"Autonomous mission for: {mission.goal.objective}",
            strategy_name=strategy_name,
            steps=steps,
            metadata={"goal": mission.goal.objective},
        )

    def replan(self, mission, *, reason: str) -> MissionPlan:
        plan = self.plan(mission)
        plan.metadata["replan_reason"] = reason
        return plan

    def _model_plan(self, mission) -> list[MissionStep] | None:
        if not self._settings.ollama_enabled:
            return None
        try:
            response = self._models.infer(
                ModelRequest(
                    task_type="planning",
                    logical_model="planner",
                    required_capabilities=("planning",),
                    messages=[
                        ModelMessage(
                            role="system",
                            content=(
                                "Return JSON array of mission steps. "
                                "Each item must include kind,title,target,description,payload,expected_outcome,risk_level."
                            ),
                        ),
                        ModelMessage(role="user", content=mission.goal.objective),
                    ],
                    metadata={"component": "autonomy_planner", "mission_id": mission.mission_id},
                )
            )
            payload = json.loads(response.content)
            if not isinstance(payload, list):
                return None
            steps = []
            for index, item in enumerate(payload, start=1):
                steps.append(
                    MissionStep(
                        step_id=f"step-{index}-{uuid4().hex[:6]}",
                        kind=MissionStepKind(str(item.get("kind", MissionStepKind.REASON.value))),
                        title=str(item.get("title", f"Step {index}")),
                        description=str(item.get("description", "")),
                        target=str(item.get("target", "runtime.snapshot")),
                        payload=item.get("payload", {}) if isinstance(item.get("payload", {}), dict) else {},
                        expected_outcome=str(item.get("expected_outcome")) if item.get("expected_outcome") else None,
                        risk_level=RiskLevel(str(item.get("risk_level", RiskLevel.LOW.value))),
                        status=MissionStepStatus.PENDING,
                    )
                )
            return steps or None
        except Exception:
            return None

    def _fallback_steps(self, mission, sequence: list[MissionStepKind]) -> list[MissionStep]:
        steps: list[MissionStep] = []
        payload = mission.context.payload
        index = 1
        for kind in sequence:
            if kind == MissionStepKind.OBSERVE:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Observe runtime", description="Collect runtime and mission state.", target="runtime.snapshot", expected_outcome="Current runtime state collected.", status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.RETRIEVE:
                query = payload.get("semantic_query") or mission.goal.objective
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Retrieve context", description="Collect semantic context for the mission.", target="semantic.retrieve_context", payload={"query": query, "collection_name": payload.get("collection_name")}, expected_outcome="Relevant context retrieved.", status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.ACTION and (payload.get("research_query") or any(token in mission.goal.objective.casefold() for token in ("investiga", "research", "analiza", "compara"))):
                research_query = str(payload.get("research_query") or mission.goal.objective)
                research_budget = payload.get("budget", {}) if isinstance(payload.get("budget", {}), dict) else {}
                max_sources = int(research_budget.get("max_sources", 6))
                requires_approval = bool(max_sources >= 8 or payload.get("paths") or payload.get("image_paths"))
                steps.append(
                    MissionStep(
                        step_id=f"step-{index}",
                        kind=kind,
                        title="Run deep research",
                        description="Execute deep research workflow and generate a report.",
                        target="research.run_task",
                        payload={
                            "query": research_query,
                            "task_id": payload.get("research_task_id"),
                            "collection_name": payload.get("collection_name"),
                            "paths": payload.get("paths", []),
                            "image_paths": payload.get("image_paths", []),
                            "source_scope": tuple(payload.get("source_scope", ("semantic_memory", "workspace", "simulated"))),
                            "run_via_autonomy": False,
                            "budget": research_budget,
                        },
                        expected_outcome="Research report generated.",
                        risk_level=RiskLevel.MEDIUM if requires_approval else RiskLevel.LOW,
                        requires_approval=requires_approval,
                        approval_reason="research task exceeds lightweight threshold" if requires_approval else None,
                        status=MissionStepStatus.PENDING,
                    )
                )
            elif kind == MissionStepKind.ACTION and (payload.get("writing_prompt") or any(token in mission.goal.objective.casefold() for token in ("continúa", "sigue donde", "escribe como", "continúa la escena", "writing"))):
                writing_prompt = str(payload.get("writing_prompt") or mission.goal.objective)
                desired_words = int(payload.get("desired_words", 160))
                target_window = payload.get("target_window")
                requires_approval = bool(desired_words > 250 or target_window is None)
                steps.append(
                    MissionStep(
                        step_id=f"step-{index}",
                        kind=kind,
                        title="Run writing continuation",
                        description="Continue writing in the target document while preserving style and context.",
                        target="writing.continue_task",
                        payload={
                            "prompt": writing_prompt,
                            "instruction": payload.get("instruction"),
                            "mode": "copilot",
                            "target_window": target_window,
                            "ensure_window_contains": payload.get("ensure_window_contains"),
                            "desired_words": desired_words,
                            "collection_name": payload.get("collection_name"),
                            "write_directly": True,
                        },
                        expected_outcome="Writing continuation inserted into the active document.",
                        risk_level=RiskLevel.MEDIUM if requires_approval else RiskLevel.LOW,
                        requires_approval=requires_approval,
                        approval_reason="writing task requires explicit target confirmation" if requires_approval else None,
                        status=MissionStepStatus.PENDING,
                    )
                )
            elif kind == MissionStepKind.REASON:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Interpret objective", description="Summarize the immediate objective and constraints.", target="model.chat", payload={"prompt": mission.goal.objective, "task_type": "assistant", "logical_model": "general_assistant"}, expected_outcome="Task interpretation ready.", status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.VISION:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Inspect screen", description="Observe visible UI and text.", target="vision.ui_awareness", payload={"capture": {"target_type": "active_window"}}, expected_outcome="Visible state collected.", status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.UI:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Perform UI action", description="Execute requested UI interaction.", target="interface.write_text" if "text" in payload else "interface.active_window", payload={"text": payload.get("text", mission.goal.objective), "mode": payload.get("mode", "copilot"), "target_window": payload.get("target_window")} if "text" in payload or payload.get("target_window") else {}, expected_outcome="Requested UI interaction performed.", risk_level=RiskLevel.MEDIUM, status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.VOICE:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Perform voice action", description="Use voice runtime if required by the goal.", target="voice_runtime.speak", payload={"text": payload.get("speak_text", mission.goal.objective)}, expected_outcome="Voice response issued.", risk_level=RiskLevel.MEDIUM, status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.VERIFY:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Verify objective", description="Verify the latest step outcome against the mission goal.", target="mission.verify", expected_outcome="Mission progress verified.", status=MissionStepStatus.PENDING))
            elif kind == MissionStepKind.REFLECT:
                steps.append(MissionStep(step_id=f"step-{index}", kind=kind, title="Reflect and decide", description="Decide whether to stop, retry or replan.", target="mission.reflect", expected_outcome="Next mission decision available.", status=MissionStepStatus.PENDING))
            index += 1
        return steps
