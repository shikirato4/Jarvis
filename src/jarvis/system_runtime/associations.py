from __future__ import annotations

from .base import AssociationResolution, ResolvedSystemTarget, SystemTargetKind


class AssociationResolver:
    def __init__(self, association_registry, *, logger=None) -> None:
        self._association_registry = association_registry
        self._logger = logger

    def resolve(self, target: ResolvedSystemTarget) -> AssociationResolution | None:
        for provider in self._association_registry.list_providers():
            resolved = provider.resolve(target)
            if resolved is not None:
                return resolved
        if target.kind == SystemTargetKind.FOLDER:
            return AssociationResolution(target_kind=target.kind, handler_kind="system_explorer", handler_name="system explorer", supports_open=True, supports_reveal=True)
        if target.kind == SystemTargetKind.FILE:
            return AssociationResolution(target_kind=target.kind, handler_kind="system_association", handler_name="default", supports_open=True, supports_reveal=True)
        if target.kind == SystemTargetKind.URI:
            return AssociationResolution(target_kind=target.kind, handler_kind="uri_association", handler_name="default", supports_open=True, supports_reveal=False)
        if target.kind == SystemTargetKind.APPLICATION:
            return AssociationResolution(target_kind=target.kind, handler_kind="executable", handler_name=target.display_name, handler_path=target.path, supports_open=True, supports_reveal=False)
        return None
