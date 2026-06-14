from __future__ import annotations

from jarvis.code_agent_runtime.agent.models import AgentRunMode, AgentTask, ExecutionPlan, PlanStep
from jarvis.code_agent_runtime.base import RiskLevel


class AgentPlanner:
    def build_task(self, task: str, *, skill_ids: list[str], max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> AgentTask:
        text = " ".join(task.split())[:1000]
        folded = text.casefold()
        needs_file_changes = any(token in folded for token in ("arregla", "fix", "implementa", "crear", "edita", "actualiza", "corrige", "add ", "change"))
        needs_commands = any(token in folded for token in ("pytest", "test", "build", "compile", "npm", "pip", "install"))
        needs_checkpoint = needs_file_changes
        risk = RiskLevel.SENSITIVE if any(token in folded for token in ("install", "borrar", "delete", "remove", "reset", "clean")) else RiskLevel.MINOR_CHANGE if needs_file_changes or needs_commands else RiskLevel.SAFE
        return AgentTask(
            original_text=text,
            objective=text,
            task_type=self._task_type(folded, skill_ids),
            skills=skill_ids,
            risk_level=risk,
            needs_file_changes=needs_file_changes,
            needs_commands=needs_commands,
            needs_git_checkpoint=needs_checkpoint,
            requires_confirmation=risk >= RiskLevel.SENSITIVE,
            requires_pin=risk >= RiskLevel.CRITICAL or "install" in folded,
            max_steps=max_steps,
            max_commands=max_commands,
            max_files_edited=max_files_edited,
        )

    def plan(self, task: AgentTask, *, mode: AgentRunMode = AgentRunMode.DRY_RUN) -> ExecutionPlan:
        steps: list[PlanStep] = [
            PlanStep(id="step-1", description="Build sanitized task context from memory, skills, Git, local search and project scan.", action_type="context", tool="agent_context_builder"),
            PlanStep(id="step-2", description="Review local Git status before any code change.", action_type="git_status", tool="git_manager"),
            PlanStep(id="step-3", description="Search local project files for likely relevant code.", action_type="project_search", tool="project_search", data={"query": task.objective}),
        ]
        if task.needs_local_search:
            steps.append(PlanStep(id=f"step-{len(steps)+1}", description="Search local repository library and learned patterns for reference context.", action_type="local_search", tool="local_search", data={"query": task.objective, "skills": task.skills}))
        if task.needs_commands:
            steps.append(self._command_step(task, len(steps) + 1))
        if task.needs_git_checkpoint:
            steps.append(
                PlanStep(
                    id=f"step-{len(steps)+1}",
                    description="Prepare a Git checkpoint before applying code changes.",
                    action_type="git_checkpoint",
                    tool="git_manager",
                    risk_level=RiskLevel.SENSITIVE,
                    requires_confirmation=True,
                )
            )
        if task.needs_file_changes:
            steps.append(
                PlanStep(
                    id=f"step-{len(steps)+1}",
                    description="Apply minimal file edits through CodeAgentExecutor only after explicit file targets are known.",
                    action_type="file_edit",
                    tool="file_writer",
                    risk_level=RiskLevel.MINOR_CHANGE,
                )
            )
        steps.extend(
            [
                PlanStep(id=f"step-{len(steps)+1}", description="Review final Git diff summary.", action_type="git_diff", tool="git_manager"),
                PlanStep(id=f"step-{len(steps)+2}", description="Verify result and produce final summary.", action_type="verify", tool="agent_verifier"),
                PlanStep(id=f"step-{len(steps)+3}", description="Record task, plan and outcome in project memory.", action_type="memory_record", tool="project_memory"),
            ]
        )
        limited = steps[: task.max_steps]
        warnings = []
        if len(steps) > len(limited):
            warnings.append(f"plan truncated to max_steps={task.max_steps}")
        return ExecutionPlan(task=task, mode=mode, steps=limited, warnings=warnings)

    @staticmethod
    def _command_step(task: AgentTask, number: int) -> PlanStep:
        command = "python -m pytest" if "python" in task.skills or "testing" in task.skills else "npm test"
        if "compile" in task.original_text.casefold():
            command = "python -m compileall -q src tests"
        if "install" in task.original_text.casefold():
            command = "npm install" if "npm" in task.original_text.casefold() else "pip install"
        sensitive = "install" in command
        return PlanStep(
            id=f"step-{number}",
            description=f"Run verification command if permitted: {command}",
            action_type="run_command",
            tool="terminal_runner",
            risk_level=RiskLevel.SENSITIVE if sensitive else RiskLevel.MINOR_CHANGE,
            requires_confirmation=sensitive,
            requires_pin=sensitive,
            data={"command": command},
        )

    @staticmethod
    def _task_type(text: str, skill_ids: list[str]) -> str:
        if "security-audit" in skill_ids or any(token in text for token in ("security", "permiso", "path traversal", "command injection")):
            return "security-audit"
        if "cli" in skill_ids or "cli" in text:
            return "cli"
        if "testing" in skill_ids or "pytest" in text or "test" in text:
            return "testing"
        if "frontend-react" in skill_ids:
            return "frontend-react"
        return "programming"
