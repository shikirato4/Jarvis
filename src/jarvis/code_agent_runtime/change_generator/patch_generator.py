from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.code_agent_runtime.change_generator.change_planner import ChangePlanner
from jarvis.code_agent_runtime.change_generator.models import ChangeGenerationResult, GeneratedChangePlan, GeneratedPatchProposal
from jarvis.code_agent_runtime.change_generator.target_resolver import TargetResolver
from jarvis.code_agent_runtime.llm.prompt_builder import LLMPromptBuilder
from jarvis.code_agent_runtime.llm.response_parser import LLMResponseParser

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


class ChangeGenerator:
    def __init__(self, executor: "CodeAgentExecutor") -> None:
        self._executor = executor
        self._resolver = TargetResolver(executor)
        self._planner = ChangePlanner(executor)
        self._prompt_builder = LLMPromptBuilder(executor)
        self._response_parser = LLMResponseParser(executor.project_root)

    def targets(self, task: str, *, max_targets: int = 3) -> dict:
        safe_task = self._executor.project_memory.sanitize_text(task)
        targets, reasons, warnings = self._resolver.resolve(task, max_targets=max_targets)
        status = "blocked" if any(target.blocked for target in targets) else "needs_review" if not targets or len(targets) > 1 else "resolved"
        return self._executor.project_memory.sanitize_value(
            {
                "task": safe_task,
                "status": status,
                "targets": [target.model_dump(mode="json") for target in targets],
                "reasons": reasons,
                "warnings": warnings,
            }
        )

    def plan(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False) -> dict:
        plan = self._build_llm_plan(task, max_targets=max_targets, llm_mode=llm_mode, allow_online=allow_online) if llm_assisted else self._build_plan(task, max_targets=max_targets)
        return self._executor.project_memory.sanitize_value(plan.model_dump(mode="json"))

    def propose(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False) -> dict:
        safe_task = self._executor.project_memory.sanitize_text(task)
        plan = self._build_llm_plan(task, max_targets=max_targets, llm_mode=llm_mode, allow_online=allow_online) if llm_assisted else self._build_plan(task, max_targets=max_targets)
        patch: GeneratedPatchProposal | None = None
        status = plan.status
        reasons = list(plan.reasons)
        warnings = list(plan.warnings)

        if plan.status == "proposed" and plan.operations:
            patch_payload = self._propose_patch(plan)
            patch = GeneratedPatchProposal(
                patch_id=str(patch_payload.get("id", "")),
                status=str(patch_payload.get("status", "")),
                summary=str(patch_payload.get("summary", patch_payload.get("message", ""))),
                target_files=[str(item) for item in patch_payload.get("target_files", [])],
                unified_diff=str(patch_payload.get("unified_diff", "")),
                patch=patch_payload,
            )
            status = "proposed" if patch.status == "proposed" else "blocked"
            if status == "blocked":
                reasons.append(str(patch_payload.get("message", "patch proposal was blocked")))
            warnings.extend(str(item) for item in patch_payload.get("warnings", []))

        result = ChangeGenerationResult(
            task=safe_task,
            status=status,
            skills_used=plan.skills,
            context_used=plan.context_used,
            targets=plan.targets,
            plan=plan,
            patch=patch,
            patch_id=patch.patch_id if patch else "",
            confidence=plan.confidence,
            risks=plan.risks,
            warnings=list(dict.fromkeys(warnings)),
            reasons=list(dict.fromkeys(reason for reason in reasons if reason)),
            message=self._message(status, patch),
        )
        self._record_result(result)
        return self._executor.project_memory.sanitize_value(result.model_dump(mode="json"))

    def _build_llm_plan(self, task: str, *, max_targets: int, llm_mode: str | None, allow_online: bool) -> GeneratedChangePlan:
        safe_task = self._executor.project_memory.sanitize_text(task)
        fallback = self._build_plan(task, max_targets=max_targets)
        if fallback.status == "blocked":
            fallback.context_used.append("llm_skipped_blocked_target")
            return fallback
        prompt = self._prompt_builder.build(safe_task, fallback.targets, skills=fallback.skills)
        provider, route = self._executor.llm_router.route(safe_task, fallback.targets, mode=llm_mode, allow_online_override=allow_online, prompt=prompt.prompt)
        self._executor.project_memory.add_llm_event(
            provider=route.provider_name,
            model=route.model_name,
            task=safe_task,
            targets=[target.path for target in fallback.targets],
            status="routed" if route.allowed else "blocked",
            confidence=0.0,
            mode=route.mode,
            sensitivity=route.sensitivity,
            reason=route.reason,
            fallback_used=route.fallback_used,
            warning=route.warning,
        )
        if provider is None or not route.allowed:
            fallback.warnings.append(f"LLM route unavailable; deterministic fallback used: {route.reason}")
            fallback.context_used.append(f"llm_route_{route.provider_kind}_{route.sensitivity}")
            return fallback
        generated = provider.generate_change_proposal(prompt)
        self._executor.project_memory.add_llm_event(
            provider=generated.provider_name,
            model=generated.model_name,
            task=safe_task,
            targets=[target.path for target in fallback.targets],
            status=generated.status,
            confidence=0.0,
            mode=route.mode,
            sensitivity=route.sensitivity,
            reason=route.reason,
            fallback_used=route.fallback_used,
            warning="; ".join(generated.warnings + ([generated.error] if generated.error else [])),
        )
        if generated.status != "ok":
            fallback.warnings.append(f"LLM generation failed; deterministic fallback used: {generated.error}")
            fallback.context_used.append("llm_failed")
            return fallback
        plan = self._response_parser.parse(generated.content, task=safe_task, skills=fallback.skills)
        plan.warnings = list(dict.fromkeys([*plan.warnings, *generated.warnings]))
        return plan

    def _build_plan(self, task: str, *, max_targets: int) -> GeneratedChangePlan:
        safe_task = self._executor.project_memory.sanitize_text(task)
        suggestions = self._executor.skill_router.suggest(safe_task, {}, limit=5)
        skills = [item.skill.id for item in suggestions]
        targets, reasons, warnings = self._resolver.resolve(task, max_targets=max_targets)
        plan = self._planner.plan(safe_task, targets, skills=skills, context_used=["skill_router", "target_resolver"])
        plan.reasons = list(dict.fromkeys([*reasons, *plan.reasons]))
        plan.warnings = list(dict.fromkeys([*warnings, *plan.warnings]))
        return plan

    def _propose_patch(self, plan: GeneratedChangePlan) -> dict:
        operation = plan.operations[0]
        task = plan.task
        if operation.operation == "replace":
            return self._executor.patch_propose_replace(operation.file, operation.old_text, operation.new_text, task=task)
        if operation.operation == "insert_before":
            return self._executor.patch_propose_insert_before(operation.file, operation.anchor, operation.text, task=task)
        if operation.operation == "insert_after":
            return self._executor.patch_propose_insert_after(operation.file, operation.anchor, operation.text, task=task)
        if operation.operation == "append":
            return self._executor.patch_propose_append(operation.file, operation.text, task=task)
        if operation.operation == "create_file":
            return self._executor.patch_propose_create_file(operation.file, operation.content, task=task)
        return {"status": "blocked", "message": f"unsupported operation: {operation.operation}", "target_files": [operation.file]}

    def _record_result(self, result: ChangeGenerationResult) -> None:
        self._executor.project_memory.add_change_event(
            task=result.task,
            targets=[target.path for target in result.targets],
            patch_id=result.patch_id,
            status=result.status,
            skills=result.skills_used,
            provider=str(result.plan.context_used if result.plan else ""),
            confidence=result.confidence,
            warning="; ".join(result.warnings[:3]),
        )

    @staticmethod
    def _skill_id(item: dict) -> str:
        if not isinstance(item, dict):
            return ""
        if item.get("id"):
            return str(item["id"])
        if isinstance(item.get("skill"), dict) and item["skill"].get("id"):
            return str(item["skill"]["id"])
        return ""

    @staticmethod
    def _message(status: str, patch: GeneratedPatchProposal | None) -> str:
        if status == "proposed" and patch:
            return f"reviewable patch generated: {patch.patch_id}"
        if status == "needs_review":
            return "task needs review before a patch can be generated"
        if status == "blocked":
            return "change generation was blocked"
        return status
