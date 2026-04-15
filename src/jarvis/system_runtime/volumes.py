from __future__ import annotations

from pathlib import Path

from .base import KnownLocationDescriptor, VolumeDescriptor


class VolumeTopologyService:
    def __init__(self, volume_registry, settings, *, logger=None) -> None:
        self._volume_registry = volume_registry
        self._settings = settings
        self._logger = logger

    def list_volumes(self) -> list[VolumeDescriptor]:
        volumes: list[VolumeDescriptor] = []
        seen: set[str] = set()
        for provider in self._volume_registry.list_providers():
            for entry in provider.list_volumes():
                if entry.root in seen:
                    continue
                seen.add(entry.root)
                volumes.append(entry)
        for root in self._settings.resolved_system_search_roots:
            if str(root) not in seen:
                seen.add(str(root))
                volumes.append(VolumeDescriptor(name=root.drive or root.name or str(root), root=str(root), kind="configured_root"))
        return sorted(volumes, key=lambda item: item.root.casefold())

    def list_known_locations(self) -> list[KnownLocationDescriptor]:
        items: list[KnownLocationDescriptor] = []
        seen: set[str] = set()
        for provider in self._volume_registry.list_providers():
            for item in provider.list_known_locations():
                if item.location_id in seen:
                    continue
                seen.add(item.location_id)
                items.append(item)
        for key, value in self._settings.system_known_locations.items():
            if key in seen:
                continue
            items.append(KnownLocationDescriptor(location_id=key, label=key.replace("_", " ").title(), path=str(Path(value).expanduser())))
        return sorted(items, key=lambda item: item.location_id)

    def default_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()
        for root in self._settings.resolved_system_search_roots:
            if str(root) not in seen:
                seen.add(str(root))
                roots.append(root)
        for volume in self.list_volumes():
            root = Path(volume.root)
            if str(root) not in seen and root.exists():
                seen.add(str(root))
                roots.append(root)
        for location in self.list_known_locations():
            root = Path(location.path).expanduser()
            if str(root) not in seen and root.exists():
                seen.add(str(root))
                roots.append(root)
        return roots
