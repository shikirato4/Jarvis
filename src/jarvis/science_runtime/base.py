from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ScienceSolveRequest(JarvisBaseModel):
    query: str
    operation: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    generate_plot: bool = False


class ScienceSimulationRequest(JarvisBaseModel):
    query: str | None = None
    simulation_type: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    duration: float | None = None
    time_step: float | None = None
    max_points: int = 2000
    generate_plot: bool = True


class ScienceResult(JarvisBaseModel):
    kind: Literal["solve", "simulate"]
    domain: str
    operation: str
    explanation: str
    formulas: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    table: list[dict[str, float | int | str]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
