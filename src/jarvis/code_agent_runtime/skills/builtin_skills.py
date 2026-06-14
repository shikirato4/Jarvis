from __future__ import annotations

from jarvis.code_agent_runtime.skills.base import CodeAgentSkill
from jarvis.code_agent_runtime.skills.registry import SkillRegistry


def builtin_skills() -> list[CodeAgentSkill]:
    return [
        CodeAgentSkill(
            id="python",
            name="Python",
            description="Use a Python-first workflow: inspect imports, run focused tests, then compile for syntax.",
            tags=("python", "backend", "runtime"),
            task_types=("implementation", "bugfix", "maintenance"),
            file_patterns=("*.py", "pyproject.toml", "requirements.txt"),
            safe_commands=("python -m pytest", "python -m compileall -q src\\jarvis tests"),
            checklist=("Review imports and module boundaries.", "Run pytest when tests exist.", "Run compileall after code changes.", "Keep fixes scoped."),
            avoid=("Do not introduce dependencies unless necessary.", "Do not make broad rewrites without tests."),
            keywords=("python", "pytest", "pyproject", "import", "traceback", "compileall", ".py", "memory", "memoria", "persistente", "tools"),
        ),
        CodeAgentSkill(
            id="testing",
            name="Testing",
            description="Reproduce failures, add focused regression coverage, and re-run the smallest useful suite.",
            tags=("testing", "pytest", "verification"),
            task_types=("test", "regression", "verification"),
            file_patterns=("tests/*.py", "test_*.py", "pyproject.toml"),
            safe_commands=("python -m pytest -q",),
            checklist=("Detect the active test framework.", "Read the exact failure.", "Add regression tests for fixed bugs.", "Record useful and failed commands in memory."),
            avoid=("Do not claim verification without running commands.", "Do not hide failing tests."),
            keywords=("test", "tests", "pytest", "regression", "falla", "error", "verify", "validar", "verifica", "prueba", "audit", "audita", "review", "revisa", "react", "typescript", "component", "cli", "git", "diff", "status", "checkpoint"),
        ),
        CodeAgentSkill(
            id="debugging",
            name="Debugging",
            description="Reproduce the issue, trace the root cause, apply the smallest fix, and verify again.",
            tags=("debugging", "bugfix", "root-cause"),
            task_types=("bugfix", "investigation"),
            file_patterns=("*.py", "tests/*.py", "*.ts", "*.tsx"),
            safe_commands=("python -m pytest -q", "python -m compileall -q src\\jarvis tests"),
            checklist=("Reproduce before changing code when possible.", "Read stack traces and failing assertions.", "Fix root cause, not symptoms.", "Re-run failing test first."),
            avoid=("Do not guess when errors are available.", "Do not change unrelated behavior."),
            keywords=("debug", "bug", "stack trace", "traceback", "fix", "arregla", "corrige", "root cause", "causa", "memory", "memoria"),
        ),
        CodeAgentSkill(
            id="git-review",
            name="Git Review",
            description="Review local Git state, summarize changed files, and prefer checkpoints before risky edits.",
            tags=("git", "review", "diff", "checkpoint"),
            task_types=("review", "checkpoint", "change-summary"),
            file_patterns=(".git", "*"),
            safe_commands=("git status --short --branch", "git diff --stat", "git diff"),
            checklist=("Review git status.", "Review diff stat.", "List changed files.", "Suggest checkpoint before large or risky changes.", "Do not push."),
            avoid=("Do not run git push.", "Do not reset or clean destructively."),
            keywords=("git", "diff", "status", "checkpoint", "stash", "branch", "cambios", "review", "revisa"),
        ),
        CodeAgentSkill(
            id="security-audit",
            name="Security Audit",
            description="Audit permissions, path traversal, command injection, secret handling, and sensitive actions.",
            tags=("security", "permissions", "audit", "local-safety"),
            task_types=("audit", "security", "hardening"),
            file_patterns=("security/*.py", "tools/*.py", "executor.py", "permissions.py", "command_validator.py", "path_policy.py"),
            safe_commands=("python -m pytest tests\\test_code_agent_runtime.py -q",),
            checklist=("Check PermissionGate coverage.", "Test path traversal and absolute paths.", "Test command injection patterns.", "Verify secrets are redacted.", "Add regression tests."),
            avoid=("Do not log secrets.", "Do not allow direct tool bypasses.", "Do not relax critical actions."),
            keywords=("security", "seguridad", "permission", "permisos", "path traversal", "../", "command injection", "secret", "token", "pin", "audit", "audita"),
        ),
        CodeAgentSkill(
            id="cli",
            name="CLI",
            description="Preserve CLI compatibility, add clear commands, and test Typer entry points.",
            tags=("cli", "typer", "commands"),
            task_types=("cli", "interface", "command"),
            file_patterns=("cli.py", "tests/test_system_cli.py", "tests/test_*cli*.py"),
            safe_commands=("python -m pytest tests\\test_system_cli.py -q",),
            checklist=("Inspect existing command names.", "Keep output stable and JSON where existing commands use JSON.", "Add CLI tests.", "Check useful error messages."),
            avoid=("Do not break existing command paths.", "Do not add hidden side effects."),
            keywords=("cli", "command", "comando", "typer", "python -m jarvis", "terminal"),
        ),
        CodeAgentSkill(
            id="docs",
            name="Docs",
            description="Document real behavior and verified commands without inventing unsupported features.",
            tags=("docs", "readme", "documentation"),
            task_types=("documentation", "notes"),
            file_patterns=("README.md", "docs/*.md", "*.md"),
            safe_commands=(),
            checklist=("Document only implemented behavior.", "Use real command examples.", "Mention safety constraints.", "Keep examples safe."),
            avoid=("Do not document unbuilt features as complete.", "Do not include secrets."),
            keywords=("docs", "documentation", "readme", "documenta", "manual", "guia", "guía"),
        ),
        CodeAgentSkill(
            id="frontend-react",
            name="Frontend React",
            description="Use React/TypeScript workflow: inspect components, state, build errors, and focused UI tests.",
            tags=("frontend", "react", "typescript", "ui"),
            task_types=("frontend", "component", "ui"),
            file_patterns=("*.tsx", "*.ts", "package.json", "vite.config.*", "next.config.*"),
            safe_commands=("npm run build", "npm test"),
            checklist=("Inspect component boundaries.", "Check state and props.", "Run build/tests if scripts exist.", "Avoid unnecessary dependencies."),
            avoid=("Do not add broad redesigns unless requested.", "Do not install packages without confirmation."),
            keywords=("react", "typescript", "tsx", "frontend", "component", "ui", "vite", "next"),
        ),
    ]


def build_builtin_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry
