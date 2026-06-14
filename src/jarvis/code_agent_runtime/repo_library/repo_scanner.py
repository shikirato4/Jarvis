from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from jarvis.code_agent_runtime.paths import is_ignored_dir, is_inside_project, is_sensitive_path, looks_like_text_path, relative_to_root
from jarvis.code_agent_runtime.repo_library.models import RepoRecord, RepoSnippet


class RepoLibraryScanner:
    def __init__(
        self,
        *,
        max_files_per_repo: int = 220,
        max_file_bytes: int = 120_000,
        max_snippets_per_repo: int = 40,
        max_snippet_chars: int = 1200,
    ) -> None:
        self.max_files_per_repo = max_files_per_repo
        self.max_file_bytes = max_file_bytes
        self.max_snippets_per_repo = max_snippets_per_repo
        self.max_snippet_chars = max_snippet_chars

    def scan_library(self, library_root: Path, *, max_repos: int | None = None) -> list[RepoRecord]:
        root = library_root.expanduser().resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"repo library root not found: {root}")
        repos: list[RepoRecord] = []
        for child in sorted(item for item in root.iterdir() if item.is_dir()):
            if is_ignored_dir(child) or is_sensitive_path(child):
                continue
            if not is_inside_project(root, child.resolve(strict=False)):
                continue
            if self._looks_like_repo(child):
                repos.append(self.scan_repo(child))
                if max_repos and len(repos) >= max_repos:
                    break
        return repos

    def scan_repo(self, repo_root: Path) -> RepoRecord:
        root = repo_root.resolve(strict=False)
        library_root = repo_root.parent.resolve(strict=False)
        if not is_inside_project(library_root, root):
            raise PermissionError(f"repo path resolves outside library root: {repo_root}")
        repo_id = self._repo_id(root)
        files_seen = 0
        languages: dict[str, int] = {}
        key_files: list[str] = []
        structure: list[str] = []
        snippets: list[RepoSnippet] = []

        for current, dirnames, filenames in os.walk(root):
            current_path = Path(current)
            kept_dirs: list[str] = []
            for dirname in dirnames:
                child = current_path / dirname
                if is_ignored_dir(child) or is_sensitive_path(child):
                    continue
                if not is_inside_project(root, child.resolve(strict=False)):
                    continue
                kept_dirs.append(dirname)
                if len(structure) < 80:
                    structure.append(relative_to_root(root, child))
            dirnames[:] = kept_dirs

            for filename in filenames:
                path = current_path / filename
                if files_seen >= self.max_files_per_repo:
                    continue
                if not self._can_read(path):
                    continue
                files_seen += 1
                rel = relative_to_root(root, path)
                suffix = path.suffix.lower()
                if suffix:
                    languages[suffix] = languages.get(suffix, 0) + 1
                if self._is_key_file(path):
                    key_files.append(rel)
                if len(snippets) < self.max_snippets_per_repo and self._is_snippet_candidate(path, rel):
                    snippet = self._read_snippet(path)
                    if snippet:
                        snippets.append(
                            RepoSnippet(
                                repo_id=repo_id,
                                file=rel,
                                snippet=snippet,
                                language=self._language_for(path),
                                reason=self._snippet_reason(path, rel),
                            )
                        )

        readme_summary = self._read_readme_summary(root)
        frameworks = self._detect_frameworks(root, snippets, key_files)
        tags = self._tags(languages, frameworks, key_files, snippets)
        return RepoRecord(
            id=repo_id,
            name=root.name,
            path=str(root),
            languages=dict(sorted(languages.items())),
            frameworks=frameworks,
            key_files=list(dict.fromkeys(key_files))[:80],
            readme_summary=readme_summary,
            structure=list(dict.fromkeys(structure))[:80],
            tags=tags,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            snippets=snippets,
        )

    @staticmethod
    def _repo_id(root: Path) -> str:
        safe = "".join(char.lower() if char.isalnum() else "-" for char in root.name).strip("-")
        return safe or "repo"

    @staticmethod
    def _looks_like_repo(path: Path) -> bool:
        markers = (".git", "README.md", "readme.md", "package.json", "pyproject.toml", "requirements.txt", "src", "tests", "app")
        return any((path / marker).exists() for marker in markers)

    def _can_read(self, path: Path) -> bool:
        if is_sensitive_path(path) or not looks_like_text_path(path):
            return False
        if not is_inside_project(path.parent.resolve(strict=False), path.resolve(strict=False)):
            return False
        try:
            if path.stat().st_size > self.max_file_bytes:
                return False
            chunk = path.read_bytes()[:2048]
        except OSError:
            return False
        return b"\x00" not in chunk

    @staticmethod
    def _is_key_file(path: Path) -> bool:
        name = path.name.casefold()
        return name in {"readme.md", "package.json", "pyproject.toml", "requirements.txt", "tsconfig.json"} or path.suffix.lower() in {".py", ".ts", ".tsx"}

    @staticmethod
    def _is_snippet_candidate(path: Path, rel: str) -> bool:
        folded = rel.replace("\\", "/").casefold()
        name = path.name.casefold()
        if name in {"readme.md", "package.json", "pyproject.toml", "requirements.txt"}:
            return True
        return folded.startswith(("src/", "app/", "tests/")) and path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx", ".md"}

    def _read_snippet(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        text = self._sanitize_text(text.lstrip("\ufeff"))
        return text.strip()[: self.max_snippet_chars]

    def _read_readme_summary(self, root: Path) -> str:
        for name in ("README.md", "readme.md"):
            path = root / name
            if path.exists() and self._can_read(path):
                return self._read_snippet(path)[:500]
        return ""

    @staticmethod
    def _language_for(path: Path) -> str:
        return {".py": "python", ".ts": "typescript", ".tsx": "typescript-react", ".js": "javascript", ".jsx": "javascript-react", ".md": "markdown", ".json": "json", ".toml": "toml"}.get(path.suffix.lower(), path.suffix.lower().lstrip("."))

    @staticmethod
    def _snippet_reason(path: Path, rel: str) -> str:
        name = path.name.casefold()
        if name == "readme.md":
            return "readme overview"
        if name in {"package.json", "pyproject.toml", "requirements.txt"}:
            return "dependency/config metadata"
        if rel.replace("\\", "/").startswith("tests/"):
            return "test example"
        return "source example"

    def _detect_frameworks(self, root: Path, snippets: list[RepoSnippet], key_files: list[str]) -> list[str]:
        haystack = " ".join([snippet.snippet[:4000] for snippet in snippets] + key_files).casefold()
        frameworks: list[str] = []
        package_json = root / "package.json"
        if package_json.exists() and self._can_read(package_json):
            try:
                payload = json.loads(package_json.read_text(encoding="utf-8"))
                deps = {**payload.get("dependencies", {}), **payload.get("devDependencies", {})}
                haystack += " " + " ".join(str(key) for key in deps)
            except Exception:  # noqa: BLE001
                pass
        for name, tokens in {
            "pytest": ("pytest", "fixture", "assert"),
            "typer": ("typer",),
            "click": ("click",),
            "react": ("react", "tsx", "jsx"),
            "next": ("next", "next.config"),
            "vite": ("vite", "vite.config"),
            "fastapi": ("fastapi",),
            "django": ("django",),
        }.items():
            if any(token in haystack for token in tokens):
                frameworks.append(name)
        return sorted(set(frameworks))

    @staticmethod
    def _tags(languages: dict[str, int], frameworks: list[str], key_files: list[str], snippets: list[RepoSnippet]) -> list[str]:
        tags = set(frameworks)
        if ".py" in languages:
            tags.add("python")
        if ".ts" in languages or ".tsx" in languages:
            tags.add("typescript")
        if any(path.replace("\\", "/").startswith("tests/") for path in key_files) or any(snippet.file.replace("\\", "/").startswith("tests/") for snippet in snippets):
            tags.add("testing")
        if any("cli" in snippet.snippet.casefold() or "typer" in snippet.snippet.casefold() for snippet in snippets):
            tags.add("cli")
        if any("permission" in snippet.snippet.casefold() or "security" in snippet.snippet.casefold() or "auth" in snippet.snippet.casefold() for snippet in snippets):
            tags.add("security")
        return sorted(tags)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        lowered = text.casefold()
        if any(token in lowered for token in (".env", "secret", "token", "credential", "password", "private key", "api_key", "api key", "certificate", "id_rsa")):
            return "[redacted]"
        return text
