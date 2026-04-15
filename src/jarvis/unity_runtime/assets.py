from __future__ import annotations

from pathlib import Path

from .base import UnityAssetDescriptor, UnityAssetKind, UnityProjectDescriptor, UnityScriptDescriptor
from .projects import classify_asset_kind, read_meta_guid


class UnityAssetService:
    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def list_assets(self, project: UnityProjectDescriptor, *, asset_kind: UnityAssetKind | None = None) -> list[UnityAssetDescriptor]:
        root = Path(project.project_root)
        assets: list[UnityAssetDescriptor] = []
        for path in sorted((root / "Assets").rglob("*")):
            if path.name.endswith(".meta"):
                continue
            kind = classify_asset_kind(path)
            if asset_kind is not None and kind != asset_kind:
                continue
            if kind == UnityAssetKind.UNKNOWN and not path.is_dir():
                continue
            assets.append(
                UnityAssetDescriptor(
                    asset_kind=kind,
                    name=path.name,
                    asset_path=path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix(),
                    guid=read_meta_guid(path),
                )
            )
        return assets

    def search_assets(
        self,
        project: UnityProjectDescriptor,
        *,
        query: str | None = None,
        asset_kind: UnityAssetKind | None = None,
        limit: int = 20,
    ) -> list[UnityAssetDescriptor]:
        query_lower = query.casefold() if query else None
        matches: list[tuple[float, UnityAssetDescriptor]] = []
        for asset in self.list_assets(project, asset_kind=asset_kind):
            score = self._score(query_lower, asset.name, asset.asset_path)
            if query_lower is not None and score <= 0:
                continue
            matches.append((score, asset))
        matches.sort(key=lambda item: (-item[0], item[1].asset_path.casefold()))
        return [asset for _, asset in matches[:limit]]

    def list_scripts(self, project: UnityProjectDescriptor) -> list[UnityScriptDescriptor]:
        root = Path(project.project_root)
        scripts: list[UnityScriptDescriptor] = []
        for path in sorted((root / "Assets").rglob("*.cs")):
            content = path.read_text(encoding="utf-8", errors="ignore")
            namespace = _extract_namespace(content)
            class_name = _extract_class_name(content) or path.stem
            scripts.append(
                UnityScriptDescriptor(
                    class_name=class_name,
                    namespace=namespace,
                    asset_path=path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix(),
                    folder_path=path.parent.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix(),
                )
            )
        return scripts

    @staticmethod
    def _score(query_lower: str | None, name: str, asset_path: str) -> float:
        if query_lower is None:
            return 0.5
        lowered_name = name.casefold()
        lowered_path = asset_path.casefold()
        if lowered_name == query_lower:
            return 1.0
        if lowered_name.startswith(query_lower):
            return 0.9
        if query_lower in lowered_name:
            return 0.8
        if query_lower in lowered_path:
            return 0.65
        return 0.0


def _extract_namespace(content: str) -> str | None:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("namespace "):
            return line.removeprefix("namespace ").split("{", 1)[0].strip()
    return None


def _extract_class_name(content: str) -> str | None:
    tokens = content.replace("{", " { ").replace(":", " : ").split()
    for index, token in enumerate(tokens):
        if token in {"class", "struct"} and index + 1 < len(tokens):
            return tokens[index + 1]
    return None
