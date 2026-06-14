from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from jarvis.code_agent_runtime.paths import is_ignored_dir, is_sensitive_path, relative_to_root

_REDACTED = "[redacted]"
_MAX_ITEMS = 80


class ProjectMemory:
    def __init__(self, path: Path, *, project_root: Path | None = None) -> None:
        self._path = path
        self._project_root = project_root.resolve(strict=False) if project_root else None

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self, *, project_name: str | None = None, project_root: Path | None = None) -> dict:
        root = (project_root or self._project_root or Path.cwd()).resolve(strict=False)
        data = self._default_memory(project_name=project_name or root.name, project_root=root)
        self.save(data)
        return data

    def load(self) -> dict:
        if not self._path.exists():
            return self.initialize()
        try:
            return self._normalize(json.loads(self._path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return self._recover_corrupt_memory()

    def save(self, memory: dict) -> dict:
        sanitized = self._sanitize(memory)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")
        return sanitized

    def save_summary(self, summary: dict) -> None:
        memory = self.load()
        memory["project_structure"] = self._sanitize(summary)
        memory["last_activity"] = self._activity("project_structure_updated", {"summary": summary.get("summary")})
        self.save(memory)

    def load_summary(self) -> dict | None:
        if not self._path.exists():
            return None
        return self.load().get("project_structure")

    def update_project_structure(self, summary: dict) -> dict:
        memory = self.load()
        memory["project_structure"] = self._sanitize(summary)
        memory["important_files"] = self._merge_unique(memory.get("important_files", []), [item.get("path") for item in summary.get("files", []) if isinstance(item, dict)])
        memory["last_activity"] = self._activity("project_structure_updated", {"file_count": summary.get("total_files_seen")})
        return self.save(memory)

    def scan_project(self, project_root: Path | None = None) -> dict:
        root = (project_root or self._project_root or Path.cwd()).resolve(strict=False)
        directories: list[str] = []
        files: list[str] = []
        languages: dict[str, int] = {}
        scripts: dict[str, str] = {}
        dependencies: list[str] = []
        modules: list[str] = []
        for current, dirnames, filenames in self._walk(root):
            current_path = Path(current)
            directories.extend(relative_to_root(root, current_path / dirname) for dirname in dirnames[:20])
            for filename in filenames:
                path = current_path / filename
                if is_sensitive_path(path):
                    continue
                rel = relative_to_root(root, path)
                if self._is_key_file(path):
                    files.append(rel)
                suffix = path.suffix.lower()
                if suffix:
                    languages[suffix] = languages.get(suffix, 0) + 1
                if path.name == "package.json":
                    scripts, dependencies = self._read_package_json(path)
            if current_path.name in {"code_agent_runtime", "system_runtime", "desktop_agent_runtime", "voice_runtime", "memory", "services"}:
                modules.append(relative_to_root(root, current_path))
        summary = {
            "root": str(root),
            "top_directories": sorted(set(directories))[:60],
            "key_files": sorted(set(files))[:80],
            "languages": dict(sorted(languages.items())),
            "scripts": scripts,
            "dependencies": dependencies[:40],
            "internal_modules": sorted(set(modules))[:40],
            "scanned_at": self._now(),
        }
        memory = self.load()
        memory["project_structure"] = self._sanitize(summary)
        memory["important_files"] = self._merge_unique(memory.get("important_files", []), summary["key_files"])
        memory["last_activity"] = self._activity("project_memory_scan", {"root": str(root)})
        self.save(memory)
        return summary

    def add_decision(self, title: str, description: str, files: list[str] | None = None) -> dict:
        return self._append("technical_decisions", {"title": title, "description": description, "files": files or []})

    def add_fixed_issue(self, summary: str, cause: str, fix: str, files: list[str] | None = None) -> dict:
        return self._append("fixed_issues", {"summary": summary, "cause": cause, "fix": fix, "files": files or []})

    def add_useful_command(self, command: str, result: str, risk_level: int = 0) -> dict:
        return self._append("useful_commands", {"command": command, "result": result, "risk_level": risk_level})

    def add_failed_command(self, command: str, error: str, risk_level: int = 0) -> dict:
        return self._append("failed_commands", {"command": command, "error": error, "risk_level": risk_level})

    def add_pending_task(self, title: str, description: str = "", priority: str = "normal") -> dict:
        return self._append("pending_tasks", {"title": title, "description": description, "priority": priority, "status": "pending"})

    def complete_task(self, task_id: str) -> dict:
        memory = self.load()
        for task in memory.get("pending_tasks", []):
            if task.get("id") == task_id:
                task["status"] = "completed"
                task["completed_at"] = self._now()
                memory["last_activity"] = self._activity("task_completed", {"task_id": task_id})
                return self.save(memory)
        raise KeyError(f"pending task not found: {task_id}")

    def add_phase_completed(self, name: str, notes: str = "") -> dict:
        return self._append("completed_phases", {"name": name, "notes": notes})

    def add_note(self, text: str) -> dict:
        return self._append("user_notes", {"text": text})

    def add_skill_suggestion(self, task: str, skill_ids: list[str]) -> dict:
        return self._append("skill_suggestions", {"task": task, "skills": skill_ids})

    def add_skill_context(self, task: str, skill_ids: list[str]) -> dict:
        return self._append("skill_contexts", {"task": task, "skills": skill_ids})

    def add_repo_library_index(self, library_root: str, repo_count: int, snippet_count: int) -> dict:
        return self._append("repo_library_indexes", {"library_root": library_root, "repo_count": repo_count, "snippet_count": snippet_count})

    def add_repo_library_search(self, query: str, skill_ids: list[str], repo_ids: list[str]) -> dict:
        return self._append("repo_library_searches", {"query": query, "skills": skill_ids, "repos": repo_ids[:20]})

    def add_repo_learning_event(self, action: str, repo_ids: list[str] | None = None, skill_ids: list[str] | None = None, count: int = 0, warning: str = "") -> dict:
        return self._append("repo_learning_events", {"action": action, "repos": repo_ids or [], "skills": skill_ids or [], "count": count, "warning": warning})

    def add_local_search_event(self, action: str, query: str = "", skill_ids: list[str] | None = None, count: int = 0, warning: str = "") -> dict:
        return self._append("local_search_events", {"action": action, "query": query, "skills": skill_ids or [], "count": count, "warning": warning})

    def add_agent_event(
        self,
        action: str,
        task: str,
        mode: str,
        skills: list[str] | None = None,
        status: str = "",
        touched_files: list[str] | None = None,
        commands: list[str] | None = None,
        plan_steps: list[dict] | None = None,
        search_count: int = 0,
        warning: str = "",
    ) -> dict:
        return self._append(
            "agent_events",
            {
                "action": action,
                "task": task,
                "mode": mode,
                "skills": skills or [],
                "status": status,
                "touched_files": (touched_files or [])[:20],
                "commands": (commands or [])[:10],
                "plan_steps": (plan_steps or [])[:20],
                "search_count": search_count,
                "warning": warning,
            },
        )

    def add_patch_event(self, action: str, patch_id: str = "", files: list[str] | None = None, summary: str = "", status: str = "", skills: list[str] | None = None, commands: list[str] | None = None, warning: str = "") -> dict:
        return self._append(
            "patch_events",
            {
                "action": action,
                "patch_id": patch_id,
                "files": (files or [])[:20],
                "summary": summary,
                "status": status,
                "skills": skills or [],
                "commands": (commands or [])[:10],
                "warning": warning,
            },
        )

    def add_change_event(self, task: str, targets: list[str] | None = None, patch_id: str = "", status: str = "", skills: list[str] | None = None, provider: str = "", confidence: float = 0.0, warning: str = "") -> dict:
        return self._append(
            "change_events",
            {
                "task": task,
                "targets": (targets or [])[:20],
                "patch_id": patch_id,
                "status": status,
                "skills": skills or [],
                "provider": provider,
                "confidence": confidence,
                "warning": warning,
            },
        )

    def add_llm_event(self, provider: str, model: str, task: str, targets: list[str] | None = None, status: str = "", confidence: float = 0.0, mode: str = "", sensitivity: str = "", reason: str = "", fallback_used: bool = False, warning: str = "") -> dict:
        return self._append(
            "llm_events",
            {
                "provider": provider,
                "model": model,
                "task": task,
                "targets": (targets or [])[:20],
                "status": status,
                "confidence": confidence,
                "mode": mode,
                "sensitivity": sensitivity,
                "reason": reason,
                "fallback_used": fallback_used,
                "warning": warning,
            },
        )

    def sanitize_text(self, value: str) -> str:
        return self._redact(value)

    def sanitize_value(self, value):
        return self._sanitize(value)

    def compact(self, *, max_items: int = _MAX_ITEMS) -> dict:
        memory = self.load()
        for key, value in list(memory.items()):
            if isinstance(value, list) and len(value) > max_items:
                memory[key] = value[-max_items:]
        memory["last_activity"] = self._activity("memory_compacted", {"max_items": max_items})
        return self.save(memory)

    def get_agent_context_summary(self, max_chars: int = 4000) -> str:
        memory = self.load()
        pending = [task for task in memory.get("pending_tasks", []) if task.get("status") != "completed"]
        root = Path(str(memory.get("project_root", "")))
        sections = [
            f"Project: {memory.get('project_name')} at {root.name or memory.get('project_root')}",
            f"Completed phases: {', '.join(item.get('name', '') for item in memory.get('completed_phases', [])[-6:]) or 'none recorded'}",
            f"Important modules/files: {', '.join(memory.get('important_files', [])[:12]) or 'none recorded'}",
            f"Security rules: {', '.join(memory.get('security_rules', [])[:8])}",
            f"Pending tasks: {self._summaries(pending, 'title')}",
            f"Recent decisions: {self._summaries(memory.get('technical_decisions', []), 'title')}",
            f"Fixed issues: {self._summaries(memory.get('fixed_issues', []), 'summary')}",
            f"Useful commands: {self._summaries(memory.get('useful_commands', []), 'command')}",
            f"Last activity: {memory.get('last_activity')}",
        ]
        return "\n".join(sections)[:max_chars]

    def _append(self, key: str, payload: dict) -> dict:
        memory = self.load()
        item = self._sanitize(payload | {"id": self._new_id(key), "created_at": self._now()})
        memory.setdefault(key, []).append(item)
        if len(memory[key]) > _MAX_ITEMS:
            memory[key] = memory[key][-_MAX_ITEMS:]
        memory["last_activity"] = self._activity(f"{key}_updated", {"id": item["id"]})
        return self.save(memory)

    def _default_memory(self, *, project_name: str, project_root: Path) -> dict:
        return {
            "schema_version": 1,
            "project_name": self._redact(project_name),
            "project_root": str(project_root),
            "project_structure": {},
            "important_files": [],
            "completed_phases": [],
            "technical_decisions": [],
            "fixed_issues": [],
            "useful_commands": [],
            "failed_commands": [],
            "pending_tasks": [],
            "skill_suggestions": [],
            "skill_contexts": [],
            "repo_library_indexes": [],
            "repo_library_searches": [],
            "repo_learning_events": [],
            "local_search_events": [],
            "agent_events": [],
            "patch_events": [],
            "change_events": [],
            "llm_events": [],
            "security_rules": [
                "Do not read or store .env, credentials, keys, tokens or private certificates.",
                "Sensitive actions must pass through CodeAgentExecutor and PermissionGate.",
                "Dangerous commands must remain blocked or require explicit admin approval.",
                "Do not act outside the project root.",
            ],
            "user_notes": [],
            "warnings": [],
            "last_activity": self._activity("memory_initialized", {"project_root": str(project_root)}),
        }

    def _normalize(self, memory: dict) -> dict:
        default = self._default_memory(project_name=(self._project_root.name if self._project_root else "project"), project_root=self._project_root or Path.cwd())
        merged = default | memory
        for key, value in default.items():
            if isinstance(value, list):
                merged.setdefault(key, [])
        return self._sanitize(merged)

    def _recover_corrupt_memory(self) -> dict:
        backup_path = self._corrupt_backup_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            copy2(self._path, backup_path)
        memory = self._default_memory(project_name=(self._project_root.name if self._project_root else "project"), project_root=self._project_root or Path.cwd())
        memory["warnings"].append(
            {
                "id": self._new_id("warn"),
                "created_at": self._now(),
                "message": "project memory was corrupt and was reset",
                "backup_path": str(backup_path),
            }
        )
        memory["last_activity"] = self._activity("memory_recovered_from_corrupt_json", {"backup_path": str(backup_path)})
        return self.save(memory)

    def _sanitize(self, value):
        if isinstance(value, dict):
            return {str(key): self._sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str):
            return self._redact(value)
        return value

    @staticmethod
    def _redact(value: str) -> str:
        folded = value.casefold()
        if folded in {"public", "internal", "sensitive", "secret"}:
            return value
        if any(
            token in folded
            for token in (
                ".env",
                "secret",
                "token",
                "credential",
                "password",
                "private",
                "apikey",
                "api_key",
                "api key",
                "certificate",
                "cert",
                ".pem",
                ".key",
                "id_rsa",
            )
        ):
            return _REDACTED
        if any(pattern in folded for pattern in ("rm -rf", "del /s", "rmdir /s", "rd /s", "format ", "shutdown", "sudo ", "| sh", "| bash")):
            return _REDACTED
        return value

    def _corrupt_backup_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self._path.with_name(f"{self._path.name}.corrupt-{timestamp}.bak")

    def _walk(self, root: Path):
        import os

        for current, dirnames, filenames in os.walk(root):
            current_path = Path(current)
            dirnames[:] = [dirname for dirname in dirnames if not is_ignored_dir(current_path / dirname)]
            yield current, dirnames, filenames

    @staticmethod
    def _read_package_json(path: Path) -> tuple[dict[str, str], list[str]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}, []
        scripts = payload.get("scripts") if isinstance(payload.get("scripts"), dict) else {}
        deps = []
        for key in ("dependencies", "devDependencies"):
            section = payload.get(key)
            if isinstance(section, dict):
                deps.extend(section.keys())
        return {str(key): str(value) for key, value in scripts.items()}, sorted(set(str(item) for item in deps))

    @staticmethod
    def _is_key_file(path: Path) -> bool:
        name = path.name.casefold()
        return name in {"readme.md", "pyproject.toml", "package.json", "requirements.txt", "tsconfig.json"} or path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".md"}

    @staticmethod
    def _merge_unique(existing: list, new_items: list) -> list:
        values = [str(item) for item in [*existing, *new_items] if item]
        return list(dict.fromkeys(values))[:_MAX_ITEMS]

    @staticmethod
    def _summaries(items: list[dict], field: str) -> str:
        return "; ".join(str(item.get(field, "")) for item in items[-5:] if item.get(field)) or "none recorded"

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix[:4]}-{uuid4().hex[:8]}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _activity(self, action: str, payload: dict) -> dict:
        return {"action": action, "at": self._now(), "payload": self._sanitize(payload)}
