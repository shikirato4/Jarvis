from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.agent_runtime.prompt_policy import SKILLS_MARKER, AgenticPromptPolicy
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage
from jarvis.skills_runtime.models import Skill
from jarvis.skills_runtime.registry import SkillRegistry
from jarvis.skills_runtime.service import SkillService


def _write_skill(directory: Path, filename: str, body: str, *, extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        "---\n"
        "name: research-web\n"
        "description: Investigacion web con validacion de citas\n"
        "triggers: investiga, research, fuentes, busca en la web\n"
        "version: 2\n"
        f"{extra}"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_registry_parses_frontmatter_and_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "## Procedimiento\nUsa research tools y valida citas.")

    catalog = SkillRegistry().load_directory(tmp_path)

    assert len(catalog.skills) == 1
    skill = catalog.skills[0]
    assert skill.name == "research-web"
    assert skill.description == "Investigacion web con validacion de citas"
    assert skill.triggers == ("investiga", "research", "fuentes", "busca en la web")
    assert skill.version == 2
    assert "valida citas" in skill.body
    assert catalog.warnings == ()


def test_registry_skips_invalid_skill_files_with_warning(tmp_path: Path) -> None:
    (tmp_path / "sin_frontmatter.md").write_text("Solo cuerpo, sin metadatos.", encoding="utf-8")
    (tmp_path / "clave_desconocida.md").write_text(
        "---\nname: x\ndescription: y\npirate_flag: arrr\n---\ncuerpo",
        encoding="utf-8",
    )
    (tmp_path / "incompleto.md").write_text(
        "---\nname: solo-nombre\n---\ncuerpo",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("no es markdown", encoding="utf-8")

    catalog = SkillRegistry().load_directory(tmp_path)

    assert catalog.skills == ()
    assert len(catalog.warnings) == 3
    assert all("skills" in warning.lower() or "skill" in warning.lower() for warning in catalog.warnings)


def test_registry_missing_directory_returns_empty_catalog_with_warning(tmp_path: Path) -> None:
    catalog = SkillRegistry().load_directory(tmp_path / "no-existe")

    assert catalog.skills == ()
    assert len(catalog.warnings) == 1
    assert "no-existe" in catalog.warnings[0]


def test_service_selects_skills_by_trigger_relevance(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "Investiga con citas.")
    windows = (
        "---\n"
        "name: windows-desktop\n"
        "description: Operaciones de escritorio Windows\n"
        "triggers: abre notepad, lanza aplicacion\n"
        "---\n"
        "Confirma antes de lanzar."
    )
    (tmp_path / "windows.md").write_text(windows, encoding="utf-8")
    service = SkillService(SkillRegistry().load_directory(tmp_path))

    research = service.select_for_query("investiga el precio del oro, revisa fuentes")
    windows_selected = service.select_for_query("abre notepad y escribe algo")
    none_selected = service.select_for_query("hola, como estas?")

    assert [skill.name for skill in research] == ["research-web"]
    assert [skill.name for skill in windows_selected] == ["windows-desktop"]
    assert none_selected == []


def test_service_context_renders_only_matching_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "Usa research tools y valida cada cita.")
    service = SkillService(SkillRegistry().load_directory(tmp_path))

    context = service.skills_context("investiga sobre energia solar")
    empty = service.skills_context("buenos dias")

    assert "research-web" in context
    assert "Usa research tools" in context
    assert empty == ""


def test_service_context_truncates_long_bodies(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "A" * 5000)
    service = SkillService(SkillRegistry().load_directory(tmp_path))

    context = service.skills_context("investiga algo")

    assert "research-web" in context
    assert len(context) < 3_000
    assert "truncated" in context.lower()


def test_prompt_policy_injects_matching_skills_layer(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "Usa research tools y valida cada cita.")
    service = SkillService(SkillRegistry().load_directory(tmp_path))
    policy = AgenticPromptPolicy(skills_provider=service.skills_context)
    messages = [
        ModelMessage(role="system", content="Eres JARVIS."),
        ModelMessage(role="user", content="investiga el precio del oro"),
    ]

    composed = policy.compose(messages, [])
    system_content = composed[0].content

    assert SKILLS_MARKER in system_content
    assert "research-web" in system_content
    # Idempotency: recomposing does not duplicate the skills layer.
    recomposed = policy.compose(composed, [])
    assert recomposed[0].content.count(SKILLS_MARKER) == 1


def test_prompt_policy_omits_skills_layer_when_nothing_matches(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research.md", "Usa research tools.")
    service = SkillService(SkillRegistry().load_directory(tmp_path))
    policy = AgenticPromptPolicy(skills_provider=service.skills_context)
    messages = [
        ModelMessage(role="system", content="Eres JARVIS."),
        ModelMessage(role="user", content="hola que tal"),
    ]

    composed = policy.compose(messages, [])

    assert SKILLS_MARKER not in composed[0].content


def test_prompt_policy_without_provider_keeps_working() -> None:
    policy = AgenticPromptPolicy()
    messages = [ModelMessage(role="user", content="investiga algo")]

    composed = policy.compose(messages, [])

    assert SKILLS_MARKER not in composed[0].content


def test_bootstrap_wires_skills_service(tmp_path: Path) -> None:
    from jarvis.bootstrap import build_application

    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "research.md", "Usa research tools.")
    (skills_dir / "windows.md").write_text(
        "---\nname: windows-desktop\ndescription: Escritorio\ntriggers: abre notepad\n---\nConfirma.",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        skills_dir=skills_dir,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )

    app = build_application(settings)
    try:
        status = app.skills_service.status()
        assert status["loaded"] == 2
        assert status["warnings_count"] == 0
    finally:
        app.stop()


def test_skill_model_is_immutable() -> None:
    from pydantic import ValidationError

    skill = Skill(name="x", description="y", triggers=("a",), version=1, body="z")

    with pytest.raises(ValidationError):
        skill.name = "otro"  # type: ignore[misc]
