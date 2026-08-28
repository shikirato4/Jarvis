from __future__ import annotations

from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class DataTable(JarvisBaseModel):
    table_id: str
    source_name: str
    columns: list[str]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataAnalysisError(ValueError):
    pass

