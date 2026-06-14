from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.code_agent_runtime.agent.context_builder import AgentContextBuilder
from jarvis.code_agent_runtime.agent.models import AgentRunMode, AgentRunResult, PlanStepStatus, VerificationResult
from jarvis.code_agent_runtime.agent.planner import AgentPlanner
from jarvis.code_agent_runtime.agent.verifier import AgentVerifier
from jarvis.code_agent_runtime.base import CodeActionStatus, RiskLevel

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


class AgentRunner:
    def __init__(self, executor: "CodeAgentExecutor") -> None:
        self._executor = executor
        self._planner = AgentPlanner()
        self._verifier = AgentVerifier()
        self._context_builder = AgentContextBuilder(executor)

    def context(self, task: str) -> dict:
        return self._context_builder.build(task).model_dump(mode="json")

    def plan(self, task: str, *, mode: AgentRunMode = AgentRunMode.DRY_RUN, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> dict:
        context = self._context_builder.build(task)
        skill_ids = self._skill_ids(context.model_dump(mode="json"))
        agent_task = self._planner.build_task(task, skill_ids=skill_ids, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited)
        plan = self._planner.plan(agent_task, mode=mode)
        return {"context": context.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}

    def run(self, task: str, *, mode: AgentRunMode = AgentRunMode.DRY_RUN, confirm: bool = False, pin: str | None = None, patch_id: str | None = None, generate_patch: bool = False, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> dict:
        planned = self.plan(task, mode=mode, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited)
        context = planned["context"]
        plan = self._planner.plan(
            self._planner.build_task(task, skill_ids=self._skill_ids(context), max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited),
            mode=mode,
        )
        touched_files: list[str] = []
        commands: list[str] = []
        errors: list[str] = []
        patch_payload: dict = {}
        command_count = 0
        file_edit_count = 0

        if patch_id:
            patch_payload = self._executor.patch_show(patch_id)
        elif generate_patch:
            patch_payload = self._executor.change_propose(task, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online)

        if generate_patch and not patch_id:
            for step in plan.steps:
                step.status = PlanStepStatus.SKIPPED
                step.result_summary = "generate-patch: patch proposal was generated; no agent step was executed"
            verification = self._verifier.verify(plan)
            result = AgentRunResult(
                mode=mode,
                task=plan.task,
                context=context,
                plan=plan,
                verification=verification,
                patch=patch_payload,
                errors=["generated patch was not applied automatically; review it and use code patch apply <patch-id> --confirm"] if mode == AgentRunMode.APPLY else [],
                summary="Generated a reviewable patch proposal; it was not applied.",
            )
            self._record_result(result)
            return result.model_dump(mode="json")

        if mode == AgentRunMode.DRY_RUN:
            for step in plan.steps:
                step.status = PlanStepStatus.SKIPPED
                step.result_summary = "dry-run: step was planned but not executed"
            verification = self._verifier.verify(plan)
            summary = "Dry run generated a safe execution plan."
            if generate_patch:
                summary = "Dry run generated a reviewable patch proposal; it was not applied."
            result = AgentRunResult(mode=mode, task=plan.task, context=context, plan=plan, verification=verification, patch=patch_payload, summary=summary)
            self._record_result(result)
            return result.model_dump(mode="json")

        for step in plan.steps:
            if step.risk_level >= RiskLevel.SENSITIVE and (not confirm or step.requires_pin and not pin):
                step.status = PlanStepStatus.BLOCKED
                step.result_summary = "step requires explicit confirmation/PIN before execution"
                break
            if step.action_type == "file_edit":
                file_edit_count += 1
                step.status = PlanStepStatus.BLOCKED
                step.result_summary = "file edits require explicit file targets/content; no autonomous edit was attempted"
                break
            if step.action_type == "run_command":
                if command_count >= max_commands:
                    step.status = PlanStepStatus.BLOCKED
                    step.result_summary = "max command limit reached"
                    break
                command_count += 1
            if file_edit_count > max_files_edited:
                step.status = PlanStepStatus.BLOCKED
                step.result_summary = "max file edit limit reached"
                break
            self._execute_step(step, mode=mode, confirm=confirm, pin=pin, touched_files=touched_files, commands=commands, errors=errors)
            if step.status in {PlanStepStatus.BLOCKED, PlanStepStatus.FAILED}:
                break

        if patch_id and not any(step.status in {PlanStepStatus.BLOCKED, PlanStepStatus.FAILED} for step in plan.steps):
            if mode == AgentRunMode.ASSISTED and not confirm:
                patch_payload = {"patch_id": patch_id, "status": "blocked", "message": "patch application requires explicit confirmation in assisted mode"}
                errors.append("patch application requires explicit confirmation")
            else:
                patch_payload = self._executor.patch_apply(patch_id, confirm=confirm, pin=pin)
                touched_files.extend(patch_payload.get("touched_files", []))
                commands.extend(patch_payload.get("commands", []))
                errors.extend(patch_payload.get("errors", []))

        verification = self._verifier.verify(plan)
        result = AgentRunResult(
            mode=mode,
            task=plan.task,
            context=context,
            plan=plan,
            verification=verification,
            touched_files=touched_files,
            commands=commands,
            errors=errors,
            patch=patch_payload,
            summary=self._summary(plan, verification.status),
        )
        self._record_result(result)
        return result.model_dump(mode="json")

    def verify(self) -> dict:
        diff = self._executor.git_diff_stat()
        status = "success" if diff.status == CodeActionStatus.OK else "partial"
        return VerificationResult(status=status, explanation=diff.message, next_steps=["Review git diff and run targeted tests."]).model_dump(mode="json")

    def _execute_step(self, step, *, mode: AgentRunMode, confirm: bool, pin: str | None, touched_files: list[str], commands: list[str], errors: list[str]) -> None:
        step.status = PlanStepStatus.RUNNING
        try:
            if step.action_type == "context":
                step.data = {"ready": True}
                step.result_summary = "context already built"
            elif step.action_type == "git_status":
                receipt = self._executor.git_summary()
                step.data = receipt.model_dump(mode="json")
                step.result_summary = receipt.message
            elif step.action_type == "project_search":
                receipt = self._executor.search_project(str(step.data.get("query") or self._executor.project_root.name), mode="content")
                step.data = receipt.model_dump(mode="json")
                step.result_summary = receipt.message
            elif step.action_type == "local_search":
                result = self._executor.local_search_query(str(step.data.get("query") or self._executor.project_root.name), limit=5, skill_ids=[str(item) for item in step.data.get("skills", [])])
                step.data = result
                step.result_summary = f"local search returned {result.get('result_count', 0)} results"
            elif step.action_type == "git_checkpoint":
                receipt = self._executor.git_checkpoint(confirm=confirm, pin=pin, message="jarvis agent checkpoint")
                step.data = receipt.model_dump(mode="json")
                step.result_summary = receipt.message
                commands.extend(receipt.commands)
            elif step.action_type == "run_command":
                command = str(step.data.get("command", ""))
                receipt = self._executor.run_command(command, confirm=confirm, pin=pin, dry_run=(mode == AgentRunMode.ASSISTED))
                step.data = receipt.model_dump(mode="json")
                step.result_summary = receipt.message
                commands.extend(receipt.commands)
                errors.extend(receipt.errors)
                if receipt.status in {CodeActionStatus.BLOCKED, CodeActionStatus.CONFIRMATION_REQUIRED}:
                    step.status = PlanStepStatus.BLOCKED
                    return
                if receipt.status == CodeActionStatus.FAILED:
                    step.status = PlanStepStatus.FAILED
                    return
            elif step.action_type == "git_diff":
                receipt = self._executor.git_diff_stat()
                step.data = receipt.model_dump(mode="json")
                step.result_summary = receipt.message
                commands.extend(receipt.commands)
            elif step.action_type == "verify":
                step.result_summary = "verification deferred to final verifier"
            elif step.action_type == "memory_record":
                step.result_summary = "memory event recorded at run completion"
            else:
                step.result_summary = "unknown step skipped"
            step.status = PlanStepStatus.DONE
        except Exception as exc:  # noqa: BLE001
            step.status = PlanStepStatus.FAILED
            step.result_summary = str(exc)
            errors.append(str(exc))

    def _record_result(self, result: AgentRunResult) -> None:
        self._executor.project_memory.add_agent_event(
            action="agent_run",
            task=result.task.original_text,
            mode=result.mode.value,
            skills=result.task.skills,
            status=result.verification.status,
            touched_files=result.touched_files,
            commands=result.commands,
            plan_steps=[{"id": step.id, "action_type": step.action_type, "tool": step.tool, "status": step.status.value} for step in result.plan.steps],
            search_count=sum(1 for step in result.plan.steps if step.action_type in {"local_search", "project_search"} and step.status == PlanStepStatus.DONE),
            warning="; ".join(result.errors[:3]),
        )

    @staticmethod
    def _summary(plan, status: str) -> str:
        done = sum(1 for step in plan.steps if step.status == PlanStepStatus.DONE)
        blocked = sum(1 for step in plan.steps if step.status == PlanStepStatus.BLOCKED)
        failed = sum(1 for step in plan.steps if step.status == PlanStepStatus.FAILED)
        return f"Agent run {status}: {done} done, {blocked} blocked, {failed} failed."

    @staticmethod
    def _skill_ids(context: dict) -> list[str]:
        ids: list[str] = []
        for item in context.get("skills", []):
            if not isinstance(item, dict):
                continue
            if item.get("id"):
                ids.append(str(item["id"]))
            elif isinstance(item.get("skill"), dict) and item["skill"].get("id"):
                ids.append(str(item["skill"]["id"]))
        return list(dict.fromkeys(ids))
