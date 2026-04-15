from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field, field_serializer

from jarvis.models.base import JarvisBaseModel
from jarvis.ui_automation.base import WindowInfo


class CaptureTargetType(StrEnum):
    SCREEN = "screen"
    WINDOW = "window"
    ACTIVE_WINDOW = "active_window"
    REGION = "region"


class OCRGranularity(StrEnum):
    LINE = "line"
    BLOCK = "block"
    WORD = "word"


class VisualElementKind(StrEnum):
    TEXT = "text"
    BUTTON = "button"
    INPUT = "input"
    ICON = "icon"
    MENU = "menu"
    WINDOW = "window"
    PANEL = "panel"


class UIAwarenessSource(StrEnum):
    SCREENSHOT = "screenshot"
    OCR = "ocr"
    UI_TREE = "ui_tree"
    FUSED = "fused"


class ScreenRegion(JarvisBaseModel):
    left: int
    top: int
    right: int
    bottom: int
    label: str | None = None
    confidence: float | None = None

    @property
    def width(self) -> int:
        return max(self.right - self.left, 0)

    @property
    def height(self) -> int:
        return max(self.bottom - self.top, 0)

    @property
    def area(self) -> int:
        return self.width * self.height


class ScreenCaptureRequest(JarvisBaseModel):
    target_type: CaptureTargetType = CaptureTargetType.SCREEN
    backend_name: str | None = None
    window_target: str | None = None
    region: ScreenRegion | None = None
    include_image_bytes: bool = True
    image_format: str = "PNG"
    persist: bool = False
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScreenCaptureResult(JarvisBaseModel):
    backend_name: str
    target_type: CaptureTargetType
    width: int
    height: int
    image_format: str = "PNG"
    image_bytes: bytes | None = None
    image_path: str | None = None
    window: WindowInfo | None = None
    region: ScreenRegion | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("image_bytes", when_used="json")
    def _serialize_image_bytes(self, value: bytes | None) -> None:
        return None


class OCRRequest(JarvisBaseModel):
    provider_name: str | None = None
    capture: ScreenCaptureRequest | None = None
    image_bytes: bytes | None = None
    image_path: str | None = None
    language: str | None = None
    granularity: OCRGranularity = OCRGranularity.LINE
    persist_result: bool = False
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRTextBlock(JarvisBaseModel):
    text: str
    region: ScreenRegion
    confidence: float | None = None
    granularity: OCRGranularity = OCRGranularity.LINE
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResult(JarvisBaseModel):
    provider_name: str
    provider_kind: str
    text: str
    language: str | None = None
    latency_ms: float = 0.0
    fallback_used: bool = False
    blocks: list[OCRTextBlock] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class UIAnchor(JarvisBaseModel):
    anchor_id: str
    label: str
    region: ScreenRegion
    element_kind: VisualElementKind | None = None
    source: UIAwarenessSource = UIAwarenessSource.SCREENSHOT
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualElement(JarvisBaseModel):
    element_id: str
    kind: VisualElementKind
    label: str | None = None
    text: str | None = None
    region: ScreenRegion
    confidence: float = 0.0
    source: UIAwarenessSource = UIAwarenessSource.SCREENSHOT
    actionable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionAnalysisRequest(JarvisBaseModel):
    capture: ScreenCaptureRequest | None = None
    image_bytes: bytes | None = None
    image_path: str | None = None
    analyzer_name: str | None = None
    include_ocr: bool = True
    include_ui_metadata: bool = True
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionAnalysisResult(JarvisBaseModel):
    analyzer_name: str
    summary: str
    source: UIAwarenessSource = UIAwarenessSource.SCREENSHOT
    regions: list[ScreenRegion] = Field(default_factory=list)
    elements: list[VisualElement] = Field(default_factory=list)
    anchors: list[UIAnchor] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class UIAwarenessRequest(JarvisBaseModel):
    capture: ScreenCaptureRequest | None = None
    image_bytes: bytes | None = None
    image_path: str | None = None
    analyzer_name: str | None = None
    ocr_provider_name: str | None = None
    include_ocr: bool = True
    include_ui_tree: bool = True
    include_screenshot: bool = True
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UIAwarenessResult(JarvisBaseModel):
    source: UIAwarenessSource = UIAwarenessSource.FUSED
    window: WindowInfo | None = None
    regions: list[ScreenRegion] = Field(default_factory=list)
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)
    elements: list[VisualElement] = Field(default_factory=list)
    anchors: list[UIAnchor] = Field(default_factory=list)
    summary: str = ""
    source_confidence: dict[str, float] = Field(default_factory=dict)
    fusion_warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextLocationMatch(JarvisBaseModel):
    text: str
    region: ScreenRegion
    confidence: float = 0.0
    source: UIAwarenessSource = UIAwarenessSource.OCR
    metadata: dict[str, Any] = Field(default_factory=dict)


class ElementLocationMatch(JarvisBaseModel):
    element_id: str
    kind: VisualElementKind
    label: str | None = None
    region: ScreenRegion
    confidence: float = 0.0
    source: UIAwarenessSource = UIAwarenessSource.FUSED
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextLocationRequest(JarvisBaseModel):
    text: str
    awareness: UIAwarenessRequest | None = None
    case_sensitive: bool = False
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ElementLocationRequest(JarvisBaseModel):
    kind: VisualElementKind | None = None
    label: str | None = None
    awareness: UIAwarenessRequest | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    success: bool
    message: str
    backend: str | None = None
    provider: str | None = None
    analyzer: str | None = None
    latency_ms: float | None = None
    capture_target: str | None = None
    fallback_used: bool = False
    capture_result: ScreenCaptureResult | None = None
    ocr_result: OCRResult | None = None
    analysis_result: VisionAnalysisResult | None = None
    awareness_result: UIAwarenessResult | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScreenCaptureBackend(Protocol):
    backend_name: str

    def health_check(self) -> dict[str, Any]: ...

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult: ...


class OCRProvider(Protocol):
    provider_name: str
    provider_kind: str

    def health_check(self) -> dict[str, Any]: ...

    def extract_text(self, request: OCRRequest, *, capture: ScreenCaptureResult | None = None) -> OCRResult: ...


class VisionAnalyzer(Protocol):
    analyzer_name: str

    def health_check(self) -> dict[str, Any]: ...

    def analyze(
        self,
        request: VisionAnalysisRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> VisionAnalysisResult: ...

    def build_awareness(
        self,
        request: UIAwarenessRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> UIAwarenessResult | None: ...


class UIMetadataAdapter(Protocol):
    def get_active_window(self) -> WindowInfo | None: ...

    def list_windows(self) -> list[WindowInfo]: ...
