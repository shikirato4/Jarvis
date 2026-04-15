from __future__ import annotations

import re
from pathlib import Path

from jarvis.core.errors import UnityEditorOperationError

from .base import UnityAssetKind, UnityOperationReceipt, UnityScriptDescriptor


class UnityScriptService:
    CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    NAMESPACE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    def __init__(self, settings, *, logger=None) -> None:
        self._settings = settings
        self._logger = logger

    def generate_script_content(self, request) -> str:
        self._validate_class_name(request.class_name)
        if request.namespace:
            self._validate_namespace(request.namespace)
        script_type = request.script_type.casefold()
        namespace_block_start = f"namespace {request.namespace}\n{{\n" if request.namespace else ""
        namespace_block_end = "\n}" if request.namespace else ""
        using_lines = sorted(set(["using UnityEngine;"] + request.using_directives)) if script_type in {"mono_behaviour", "scriptable_object"} else sorted(set(request.using_directives))
        if script_type == "scriptable_object":
            if "using UnityEngine;" not in using_lines:
                using_lines.insert(0, "using UnityEngine;")
            body = self._scriptable_object_body(request)
        elif script_type == "plain_class":
            body = self._plain_class_body(request)
        else:
            if "using UnityEngine;" not in using_lines:
                using_lines.insert(0, "using UnityEngine;")
            body = self._mono_behaviour_body(request)
        using_block = "\n".join(using_lines).strip()
        content = [using_block, "", namespace_block_start + body + namespace_block_end]
        return "\n".join(item for item in content if item).strip() + "\n"

    def write_script(self, project, request, *, content: str) -> tuple[UnityScriptDescriptor, list[str], bool]:
        root = Path(project.project_root).resolve(strict=False)
        folder = request.folder or self._settings.unity_default_scripts_folder
        if request.asset_path:
            target = (root / request.asset_path).resolve(strict=False)
        else:
            target_folder = (root / folder).resolve(strict=False)
            target = target_folder / f"{request.class_name or 'NewScript'}.cs"
        if not str(target).startswith(str((root / "Assets").resolve(strict=False))):
            raise UnityEditorOperationError("unity scripts must be written inside Assets/")
        target.parent.mkdir(parents=True, exist_ok=True)
        would_overwrite = target.exists()
        if would_overwrite and not request.overwrite:
            raise UnityEditorOperationError(f"script '{target}' already exists and overwrite=False")
        target.write_text(content, encoding="utf-8")
        descriptor = UnityScriptDescriptor(
            class_name=request.class_name or target.stem,
            namespace=_extract_namespace(content),
            asset_path=target.resolve(strict=False).relative_to(root).as_posix(),
            folder_path=target.parent.resolve(strict=False).relative_to(root).as_posix(),
        )
        return descriptor, [str(target.resolve(strict=False))], would_overwrite

    @classmethod
    def _validate_class_name(cls, class_name: str) -> None:
        if not cls.CLASS_RE.match(class_name):
            raise UnityEditorOperationError(f"invalid C# class name '{class_name}'")

    @classmethod
    def _validate_namespace(cls, namespace: str) -> None:
        if not cls.NAMESPACE_RE.match(namespace):
            raise UnityEditorOperationError(f"invalid C# namespace '{namespace}'")

    @staticmethod
    def _field_lines(request) -> list[str]:
        lines: list[str] = []
        for entry in request.serialized_fields:
            field_type = str(entry.get("type", "string"))
            field_name = str(entry.get("name", "field"))
            default = entry.get("default")
            lines.append("    [SerializeField]")
            suffix = f" = {default};" if default is not None else ";"
            lines.append(f"    private {field_type} {field_name}{suffix}")
            lines.append("")
        return lines

    def _mono_behaviour_body(self, request) -> str:
        base_class = request.base_class or "MonoBehaviour"
        interfaces = ", ".join(request.interfaces)
        inheritance = f" : {base_class}" + (f", {interfaces}" if interfaces else "")
        lines = [f"public sealed class {request.class_name}{inheritance}", "{", *self._field_lines(request), "    private void Start()", "    {", "    }", "", "    private void Update()", "    {", "    }", "}"]
        return "\n".join(lines)

    def _scriptable_object_body(self, request) -> str:
        base_class = request.base_class or "ScriptableObject"
        interfaces = ", ".join(request.interfaces)
        inheritance = f" : {base_class}" + (f", {interfaces}" if interfaces else "")
        lines = ["[CreateAssetMenu(fileName = \"" + request.class_name + "\", menuName = \"Jarvis/" + request.class_name + "\")]", f"public sealed class {request.class_name}{inheritance}", "{", *self._field_lines(request), "}"]
        return "\n".join(lines)

    def _plain_class_body(self, request) -> str:
        base_class = request.base_class or "object"
        interfaces = ", ".join(request.interfaces)
        inheritance = "" if base_class == "object" and not interfaces else f" : {base_class}" + (f", {interfaces}" if interfaces else "")
        return "\n".join([f"public sealed class {request.class_name}{inheritance}", "{", "}"])


def _extract_namespace(content: str) -> str | None:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("namespace "):
            return line.removeprefix("namespace ").split("{", 1)[0].strip()
    return None
