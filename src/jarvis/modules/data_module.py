from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.safety import ensure_within_roots
from jarvis.data_analyzer import DataAnalyzer


class DataLoadPayload(BaseModel):
    path: str


class DataTablePayload(BaseModel):
    table_id: str


class DataColumnPayload(DataTablePayload):
    column: str


class DataFilterPayload(DataColumnPayload):
    operator: str
    value: Any = None


class DataSortPayload(DataColumnPayload):
    descending: bool = False


class DataAggregatePayload(DataColumnPayload):
    group_by: str
    operation: str


class DataPlotPayload(DataTablePayload):
    x_column: str
    y_column: str
    kind: str = "line"


class DataModule:
    name = "data"
    description = "Safe tabular analysis for CSV, TSV, JSON and XLSX files."

    def __init__(self, analyzer: DataAnalyzer) -> None:
        self._analyzer = analyzer

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="data.load_file",
                description="Load a CSV, TSV, JSON or XLSX file for tabular analysis.",
                payload_model=DataLoadPayload,
                handler=self._load,
                tags=("data", "files"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.describe",
                description="Describe columns, types, nulls and numeric statistics.",
                payload_model=DataTablePayload,
                handler=self._describe,
                tags=("data", "statistics"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.filter",
                description="Filter rows with a typed comparison operator.",
                payload_model=DataFilterPayload,
                handler=self._filter,
                tags=("data", "filter"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.sort",
                description="Sort table rows by one column.",
                payload_model=DataSortPayload,
                handler=self._sort,
                tags=("data", "sort"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.aggregate",
                description="Aggregate a numeric column by a categorical column.",
                payload_model=DataAggregatePayload,
                handler=self._aggregate,
                tags=("data", "aggregate", "statistics"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.anomalies",
                description="Detect robust numeric outliers in a column.",
                payload_model=DataColumnPayload,
                handler=self._anomalies,
                tags=("data", "anomaly", "statistics"),
            )
        )
        registry.register(
            ActionDefinition(
                name="data.plot",
                description="Generate a real line, bar or scatter chart from table columns.",
                payload_model=DataPlotPayload,
                handler=self._plot,
                tags=("data", "plot", "artifact"),
                idempotent=False,
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="data.analysis",
                module_name=self.name,
                intent="data",
                description="Load, inspect, filter, aggregate and analyze tables.",
                action_names=(
                    "data.load_file",
                    "data.describe",
                    "data.filter",
                    "data.sort",
                    "data.aggregate",
                    "data.anomalies",
                    "data.plot",
                ),
                keywords=("csv", "tsv", "xlsx", "tabla", "datos", "anomalía"),
                mode_policy=(
                    ExecutionMode.ASSIST,
                    ExecutionMode.RESEARCH,
                    ExecutionMode.OPERATOR,
                    ExecutionMode.AUTOMATION,
                ),
                task_type="analysis",
                supports_planning=True,
            )
        )

    def _load(self, context: ActionContext, payload: DataLoadPayload) -> ActionResult:
        path = ensure_within_roots(
            payload.path,
            (context.settings.resolved_workspace_root, *context.settings.resolved_research_roots),
            "data analysis",
        )
        table = self._analyzer.load_path(path)
        return ActionResult(
            message="data table loaded",
            data={
                "table_id": table.table_id,
                "source_name": table.source_name,
                "columns": table.columns,
                "row_count": len(table.rows),
                "metadata": table.metadata,
            },
        )

    def _describe(self, _context: ActionContext, payload: DataTablePayload) -> ActionResult:
        return ActionResult(
            message="data description completed",
            data=self._analyzer.describe(payload.table_id),
        )

    def _filter(self, _context: ActionContext, payload: DataFilterPayload) -> ActionResult:
        rows = self._analyzer.filter(
            payload.table_id,
            column=payload.column,
            operator=payload.operator,
            value=payload.value,
        )
        return ActionResult(message="data filter completed", data={"rows": rows, "count": len(rows)})

    def _sort(self, _context: ActionContext, payload: DataSortPayload) -> ActionResult:
        rows = self._analyzer.sort(
            payload.table_id,
            column=payload.column,
            descending=payload.descending,
        )
        return ActionResult(message="data sort completed", data={"rows": rows, "count": len(rows)})

    def _aggregate(self, _context: ActionContext, payload: DataAggregatePayload) -> ActionResult:
        values = self._analyzer.aggregate(
            payload.table_id,
            group_by=payload.group_by,
            column=payload.column,
            operation=payload.operation,
        )
        return ActionResult(message="data aggregation completed", data={"values": values})

    def _anomalies(self, _context: ActionContext, payload: DataColumnPayload) -> ActionResult:
        rows = self._analyzer.anomalies(payload.table_id, column=payload.column)
        return ActionResult(message="data anomaly analysis completed", data={"anomalies": rows, "count": len(rows)})

    def _plot(self, _context: ActionContext, payload: DataPlotPayload) -> ActionResult:
        artifact = self._analyzer.plot(
            payload.table_id,
            x_column=payload.x_column,
            y_column=payload.y_column,
            kind=payload.kind,
        )
        return ActionResult(
            message="data plot generated",
            data={"artifact": artifact, "kind": payload.kind},
            artifacts=[artifact],
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return
