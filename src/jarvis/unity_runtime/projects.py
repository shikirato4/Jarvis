from __future__ import annotations

import json
from pathlib import Path

from .base import UnityAssetDescriptor, UnityAssetKind, UnityProjectDescriptor, UnityProjectStatus, UnitySceneDescriptor


class UnityProjectService:
    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def inspect(self, project: UnityProjectDescriptor) -> UnityProjectDescriptor:
        root = Path(project.project_root)
        scenes = [
            UnitySceneDescriptor(scene_name=path.stem, asset_path=self._to_asset_path(root, path))
            for path in sorted((root / "Assets").rglob("*.unity"))
        ]
        detected = list(project.detected_features)
        if (root / "Packages" / "manifest.json").exists() and "manifest" not in detected:
            detected.append("manifest")
        if (root / "Assets").joinpath("Editor").exists() and "editor_folder" not in detected:
            detected.append("editor_folder")
        if scenes and "scenes" not in detected:
            detected.append("scenes")
        return project.model_copy(update={"scenes": scenes, "detected_features": detected, "is_valid_project": self._is_valid(root)})

    def list_scenes(self, project: UnityProjectDescriptor) -> list[UnitySceneDescriptor]:
        return self.inspect(project).scenes

    def create(self, name: str, target_root: str, *, template: str = "3d", unity_version: str | None = None) -> UnityProjectDescriptor:
        root = Path(target_root).expanduser().resolve(strict=False) / name
        (root / "Assets").mkdir(parents=True, exist_ok=True)
        (root / "Packages").mkdir(parents=True, exist_ok=True)
        (root / "ProjectSettings").mkdir(parents=True, exist_ok=True)
        manifest = root / "Packages" / "manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({"dependencies": {}, "scopedRegistries": [], "template": template}, indent=2) + "\n", encoding="utf-8")
        if unity_version:
            version_file = root / "ProjectSettings" / "ProjectVersion.txt"
            if not version_file.exists():
                version_file.write_text(f"m_EditorVersion: {unity_version}\n", encoding="utf-8")
        project = UnityProjectDescriptor(
            project_id=str(root),
            name=name,
            project_root=str(root),
            assets_path=str(root / "Assets"),
            packages_path=str(root / "Packages"),
            project_settings_path=str(root / "ProjectSettings"),
            unity_version=unity_version,
            status=UnityProjectStatus.RESOLVED,
            is_valid_project=True,
            resolution_confidence=0.9,
            detected_features=["created", "packages_manifest"],
            metadata={"template": template},
        )
        return self.inspect(project)

    def reveal_asset_descriptor(self, project: UnityProjectDescriptor, asset_path: str) -> UnityAssetDescriptor:
        path = Path(project.project_root) / asset_path
        kind = classify_asset_kind(path)
        return UnityAssetDescriptor(asset_kind=kind, name=path.name, asset_path=asset_path, guid=read_meta_guid(path))

    @staticmethod
    def _is_valid(root: Path) -> bool:
        return (root / "Assets").is_dir() and (root / "ProjectSettings").is_dir()

    @staticmethod
    def _to_asset_path(root: Path, absolute: Path) -> str:
        return absolute.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()


def classify_asset_kind(path: Path) -> UnityAssetKind:
    suffix = path.suffix.casefold()
    mapping = {
        ".unity": UnityAssetKind.SCENE,
        ".cs": UnityAssetKind.SCRIPT,
        ".prefab": UnityAssetKind.PREFAB,
        ".mat": UnityAssetKind.MATERIAL,
        ".shader": UnityAssetKind.SHADER,
        ".asset": UnityAssetKind.SCRIPTABLE_OBJECT,
        ".anim": UnityAssetKind.ANIMATION,
        ".asmdef": UnityAssetKind.ASSEMBLY_DEFINITION,
    }
    if path.is_dir():
        return UnityAssetKind.FOLDER
    return mapping.get(suffix, UnityAssetKind.UNKNOWN)


def read_meta_guid(path: Path) -> str | None:
    meta = path.with_suffix(path.suffix + ".meta")
    if not meta.exists():
        return None
    for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("guid:"):
            return line.split(":", 1)[1].strip()
    return None
