from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageModelStatus(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    ERROR = "error"


class ImageJobStatus(StrEnum):
    PENDING = "pending"
    LOADING_MODEL = "loading_model"
    GENERATING = "generating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageGenerationRequest(JarvisBaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 768
    steps: int = 25
    cfg: float = 7.0
    seed: int | None = None
    num_images: int = 1
    output_dir: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationResult(JarvisBaseModel):
    success: bool
    output_paths: list[Path] = Field(default_factory=list)
    message: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationJob(JarvisBaseModel):
    job_id: str = Field(default_factory=lambda: f"img-{uuid4().hex[:12]}")
    prompt_original: str
    prompt_positive: str
    negative_prompt: str = ""
    model_path: Path
    width: int
    height: int
    steps: int
    cfg: float
    seed: int | None = None
    status: ImageJobStatus = ImageJobStatus.PENDING
    progress: float = 0.0
    output_paths: list[Path] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def terminal(self) -> bool:
        return self.status in {ImageJobStatus.COMPLETED, ImageJobStatus.FAILED, ImageJobStatus.CANCELLED}
