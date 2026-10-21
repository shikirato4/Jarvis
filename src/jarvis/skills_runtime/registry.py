from __future__ import annotations

from pathlib import Path

from .models import Skill, SkillCatalog

_REQUIRED_KEYS = ("name", "description")
_KNOWN_KEYS = frozenset({"name", "description", "triggers", "version"})


class SkillRegistry:
    """Loads skills from a directory of ``*.md`` files with strict frontmatter.

    Format::

        ---
        name: research-web
        description: Investigacion web con validacion de citas
        triggers: investiga, research, fuentes
        version: 2
        ---
        Body of the procedure...

    Files missing frontmatter, carrying unknown keys, or missing required
    keys are skipped with a warning instead of aborting the whole catalog.
    """

    def load_directory(self, directory: Path) -> SkillCatalog:
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            name = str(directory)
            return SkillCatalog(
                warnings=(f"skills directory not found: {name}",),
            )
        skills: list[Skill] = []
        warnings: list[str] = []
        for path in sorted(directory.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                warnings.append(f"skill unreadable '{path.name}': {exc}")
                continue
            skill = self._parse(path.name, text, warnings)
            if skill is not None:
                skills.append(skill)
        return SkillCatalog(skills=tuple(skills), warnings=tuple(warnings))

    def _parse(
        self, filename: str, text: str, warnings: list[str]
    ) -> Skill | None:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            warnings.append(f"skill '{filename}' skipped: missing frontmatter block")
            return None
        metadata: dict[str, str] = {}
        end = None
        for index in range(1, len(lines)):
            line = lines[index].strip()
            if line == "---":
                end = index
                break
            if not line:
                continue
            if ":" not in line:
                warnings.append(
                    f"skill '{filename}' skipped: malformed frontmatter line '{line}'"
                )
                return None
            key, _, value = line.partition(":")
            metadata[key.strip().casefold()] = value.strip()
        if end is None:
            warnings.append(f"skill '{filename}' skipped: unterminated frontmatter")
            return None
        unknown = sorted(set(metadata) - _KNOWN_KEYS)
        if unknown:
            warnings.append(
                f"skill '{filename}' skipped: unknown frontmatter keys {unknown}"
            )
            return None
        for required in _REQUIRED_KEYS:
            if not metadata.get(required):
                warnings.append(
                    f"skill '{filename}' skipped: missing required key '{required}'"
                )
                return None
        body = "\n".join(lines[end + 1 :]).strip()
        if not body:
            warnings.append(f"skill '{filename}' skipped: empty body")
            return None
        triggers = tuple(
            trigger.strip()
            for trigger in metadata.get("triggers", "").split(",")
            if trigger.strip()
        )
        try:
            version = int(metadata.get("version") or 1)
        except ValueError:
            warnings.append(
                f"skill '{filename}' skipped: version must be an integer"
            )
            return None
        return Skill(
            name=metadata["name"],
            description=metadata["description"],
            triggers=triggers,
            version=version,
            body=body,
        )
