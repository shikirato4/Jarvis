from __future__ import annotations


class OpsMaintenanceService:
    def __init__(self, ops_runtime) -> None:
        self._ops = ops_runtime

    def sweep(self):
        return self._ops.retention_sweep()
