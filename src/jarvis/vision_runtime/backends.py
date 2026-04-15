from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageGrab

from jarvis.core.errors import VisionRuntimeError
from jarvis.ui_automation.base import WindowInfo

from .base import (
    CaptureTargetType,
    OCRGranularity,
    OCRRequest,
    OCRResult,
    OCRTextBlock,
    ScreenCaptureRequest,
    ScreenCaptureResult,
    ScreenRegion,
    UIAnchor,
    UIAwarenessRequest,
    UIAwarenessResult,
    UIAwarenessSource,
    VisualElement,
    VisualElementKind,
    VisionAnalysisRequest,
    VisionAnalysisResult,
)


def _serialize_image(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _mock_blocks_for_target(target: CaptureTargetType, window: WindowInfo | None, region: ScreenRegion | None) -> list[OCRTextBlock]:
    label = "Desktop overview"
    if target == CaptureTargetType.ACTIVE_WINDOW:
        label = f"{window.title if window else 'Active Window'} editor"
    elif target == CaptureTargetType.WINDOW:
        label = f"{window.title if window else 'Window'}"
    elif target == CaptureTargetType.REGION:
        label = region.label or "Screen region"
    text = f"{label}\nBoton Guardar\nCampo Buscar"
    return [
        OCRTextBlock(
            text="Desktop overview" if index == 0 and target == CaptureTargetType.SCREEN else line,
            region=ScreenRegion(left=24, top=24 + (index * 28), right=260, bottom=48 + (index * 28), label=line),
            confidence=0.95,
            granularity=OCRGranularity.LINE,
        )
        for index, line in enumerate(text.splitlines())
    ]


class InMemoryScreenCaptureBackend:
    backend_name = "in_memory_screen"

    def __init__(self, ui_adapter=None) -> None:
        self._ui_adapter = ui_adapter

    def health_check(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "healthy": True}

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult:
        window = None
        if self._ui_adapter is not None and hasattr(self._ui_adapter, "get_active_window"):
            if request.target_type in {CaptureTargetType.ACTIVE_WINDOW, CaptureTargetType.WINDOW}:
                if request.target_type == CaptureTargetType.WINDOW and request.window_target and hasattr(self._ui_adapter, "list_windows"):
                    window = next(
                        (
                            item
                            for item in self._ui_adapter.list_windows()
                            if request.window_target.casefold() in item.title.casefold() or request.window_target.casefold() == item.handle.casefold()
                        ),
                        None,
                    )
                window = window or self._ui_adapter.get_active_window()
        image = Image.new("RGB", (request.region.width if request.region else 640, request.region.height if request.region else 360), color=(245, 247, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, image.width - 16, 72), fill=(30, 41, 59))
        title = window.title if window else "Jarvis Screen"
        for index, line in enumerate([title, "Boton Guardar", "Campo Buscar"]):
            draw.text((24, 26 + (index * 30)), line, fill=(255, 255, 255) if index == 0 else (15, 23, 42))
        blocks = _mock_blocks_for_target(request.target_type, window, request.region)
        metadata = dict(request.metadata)
        metadata.update({"mock_ocr_blocks": [item.model_dump(mode="json") for item in blocks], "rendered_lines": [item.text for item in blocks]})
        return ScreenCaptureResult(
            backend_name=self.backend_name,
            target_type=request.target_type,
            width=image.width,
            height=image.height,
            image_format=request.image_format,
            image_bytes=_serialize_image(image, request.image_format) if request.include_image_bytes else None,
            window=window,
            region=request.region,
            metadata=metadata,
        )


class WindowsScreenCaptureBackend:
    backend_name = "windows_screen"

    def __init__(self, ui_adapter=None) -> None:
        self._ui_adapter = ui_adapter

    def health_check(self) -> dict[str, object]:
        try:
            ImageGrab.grab(all_screens=False)
            return {"backend_name": self.backend_name, "healthy": True}
        except Exception as exc:  # noqa: BLE001
            return {"backend_name": self.backend_name, "healthy": False, "error": str(exc)}

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult:
        bbox = None
        window = None
        if request.target_type == CaptureTargetType.REGION and request.region is not None:
            bbox = (request.region.left, request.region.top, request.region.right, request.region.bottom)
        elif self._ui_adapter is not None and hasattr(self._ui_adapter, "get_active_window"):
            if request.target_type in {CaptureTargetType.ACTIVE_WINDOW, CaptureTargetType.WINDOW}:
                if request.target_type == CaptureTargetType.WINDOW and request.window_target and hasattr(self._ui_adapter, "list_windows"):
                    window = next(
                        (
                            item
                            for item in self._ui_adapter.list_windows()
                            if request.window_target.casefold() in item.title.casefold() or request.window_target.casefold() == item.handle.casefold()
                        ),
                        None,
                    )
                window = window or self._ui_adapter.get_active_window()
                if window and window.rect:
                    bbox = (window.rect["left"], window.rect["top"], window.rect["right"], window.rect["bottom"])
        try:
            image = ImageGrab.grab(bbox=bbox, all_screens=False)
        except Exception as exc:  # noqa: BLE001
            raise VisionRuntimeError(str(exc)) from exc
        return ScreenCaptureResult(
            backend_name=self.backend_name,
            target_type=request.target_type,
            width=image.width,
            height=image.height,
            image_format=request.image_format,
            image_bytes=_serialize_image(image, request.image_format) if request.include_image_bytes else None,
            window=window,
            region=request.region,
            metadata=dict(request.metadata),
        )


class MssScreenCaptureBackend:
    backend_name = "windows_mss"

    def __init__(self, ui_adapter=None) -> None:
        self._ui_adapter = ui_adapter
        try:
            import mss  # type: ignore

            self._mss_module = mss
        except Exception:  # noqa: BLE001
            self._mss_module = None

    def health_check(self) -> dict[str, object]:
        if self._mss_module is None:
            return {"backend_name": self.backend_name, "healthy": False, "error": "mss is not installed"}
        try:
            with self._mss_module.mss() as sct:
                monitor = dict(sct.monitors[1])
                shot = sct.grab(monitor)
            return {"backend_name": self.backend_name, "healthy": bool(shot.width and shot.height)}
        except Exception as exc:  # noqa: BLE001
            return {"backend_name": self.backend_name, "healthy": False, "error": str(exc)}

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult:
        if self._mss_module is None:
            raise VisionRuntimeError("mss is not installed")
        monitor = None
        window = None
        if request.target_type == CaptureTargetType.REGION and request.region is not None:
            monitor = {
                "left": int(request.region.left),
                "top": int(request.region.top),
                "width": int(request.region.width),
                "height": int(request.region.height),
            }
        elif self._ui_adapter is not None and hasattr(self._ui_adapter, "get_active_window"):
            if request.target_type in {CaptureTargetType.ACTIVE_WINDOW, CaptureTargetType.WINDOW}:
                if request.target_type == CaptureTargetType.WINDOW and request.window_target and hasattr(self._ui_adapter, "list_windows"):
                    window = next(
                        (
                            item
                            for item in self._ui_adapter.list_windows()
                            if request.window_target.casefold() in item.title.casefold() or request.window_target.casefold() == item.handle.casefold()
                        ),
                        None,
                    )
                window = window or self._ui_adapter.get_active_window()
                if window and window.rect:
                    left = int(window.rect["left"])
                    top = int(window.rect["top"])
                    right = int(window.rect["right"])
                    bottom = int(window.rect["bottom"])
                    monitor = {
                        "left": left,
                        "top": top,
                        "width": max(right - left, 1),
                        "height": max(bottom - top, 1),
                    }
        try:
            with self._mss_module.mss() as sct:
                if monitor is None:
                    monitor = dict(sct.monitors[1])
                shot = sct.grab(monitor)
        except Exception as exc:  # noqa: BLE001
            raise VisionRuntimeError(str(exc)) from exc
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return ScreenCaptureResult(
            backend_name=self.backend_name,
            target_type=request.target_type,
            width=image.width,
            height=image.height,
            image_format=request.image_format,
            image_bytes=_serialize_image(image, request.image_format) if request.include_image_bytes else None,
            window=window,
            region=request.region,
            metadata=dict(request.metadata),
        )


class InMemoryOCRProvider:
    provider_name = "in_memory_ocr"
    provider_kind = "local"

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def health_check(self) -> dict[str, object]:
        return {"provider_name": self.provider_name, "provider_kind": self.provider_kind, "healthy": not self._fail}

    def extract_text(self, request: OCRRequest, *, capture: ScreenCaptureResult | None = None) -> OCRResult:
        if self._fail:
            raise VisionRuntimeError("in-memory ocr provider forced failure")
        started = time.perf_counter()
        blocks = []
        if capture is not None:
            blocks = [OCRTextBlock.model_validate(item) for item in capture.metadata.get("mock_ocr_blocks", [])]
        if not blocks and request.image_path:
            blocks = [
                OCRTextBlock(
                    text=Path(request.image_path).stem,
                    region=ScreenRegion(left=0, top=0, right=120, bottom=24, label="file-stem"),
                    confidence=0.5,
                    granularity=request.granularity,
                )
            ]
        text = "\n".join(block.text for block in blocks).strip()
        return OCRResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text=text,
            language=request.language,
            latency_ms=(time.perf_counter() - started) * 1000,
            blocks=blocks,
            metadata={"block_count": len(blocks)},
        )


class TesseractOCRProvider:
    provider_name = "tesseract_ocr"
    provider_kind = "local"

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore

            self._pytesseract = pytesseract
        except Exception:  # noqa: BLE001
            self._pytesseract = None

    def health_check(self) -> dict[str, object]:
        return {"provider_name": self.provider_name, "provider_kind": self.provider_kind, "healthy": self._pytesseract is not None}

    def extract_text(self, request: OCRRequest, *, capture: ScreenCaptureResult | None = None) -> OCRResult:
        if self._pytesseract is None:
            raise VisionRuntimeError("pytesseract is not installed")
        if capture is None and request.image_bytes is None and request.image_path is None:
            raise VisionRuntimeError("tesseract provider requires image input")
        started = time.perf_counter()
        image = None
        if capture is not None and capture.image_bytes is not None:
            image = Image.open(io.BytesIO(capture.image_bytes))
        elif request.image_bytes is not None:
            image = Image.open(io.BytesIO(request.image_bytes))
        elif request.image_path is not None:
            image = Image.open(request.image_path)
        assert image is not None
        raw = self._pytesseract.image_to_string(image, lang=request.language or "eng")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        blocks = [
            OCRTextBlock(
                text=line,
                region=ScreenRegion(left=0, top=index * 20, right=max(len(line) * 8, 40), bottom=(index * 20) + 18, label=line),
                confidence=0.6,
                granularity=request.granularity,
            )
            for index, line in enumerate(lines)
        ]
        return OCRResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text="\n".join(lines),
            language=request.language,
            latency_ms=(time.perf_counter() - started) * 1000,
            blocks=blocks,
            metadata={"engine": "tesseract"},
        )


class HeuristicVisionAnalyzer:
    analyzer_name = "heuristic_vision"

    def health_check(self) -> dict[str, object]:
        return {"analyzer_name": self.analyzer_name, "healthy": True}

    def analyze(
        self,
        request: VisionAnalysisRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> VisionAnalysisResult:
        started = time.perf_counter()
        regions = []
        elements = []
        anchors = []
        if capture is not None:
            regions.append(ScreenRegion(left=0, top=0, right=capture.width, bottom=min(72, capture.height), label="header", confidence=0.8))
        if ocr_result is not None:
            for index, block in enumerate(ocr_result.blocks, start=1):
                kind = _infer_kind(block.text)
                elements.append(
                    VisualElement(
                        element_id=f"ocr-{index}",
                        kind=kind,
                        label=block.text,
                        text=block.text,
                        region=block.region,
                        confidence=block.confidence or 0.6,
                        source=UIAwarenessSource.OCR,
                        actionable=kind in {VisualElementKind.BUTTON, VisualElementKind.INPUT, VisualElementKind.MENU},
                    )
                )
                anchors.append(
                    UIAnchor(
                        anchor_id=f"anchor-{index}",
                        label=block.text,
                        region=block.region,
                        element_kind=kind,
                        source=UIAwarenessSource.OCR,
                        confidence=block.confidence or 0.6,
                    )
                )
        summary_parts = []
        if window is not None:
            summary_parts.append(f"Active window: {window.title}")
        if ocr_result is not None and ocr_result.text:
            summary_parts.append(f"Visible text: {ocr_result.text.replace(chr(10), '; ')}")
        return VisionAnalysisResult(
            analyzer_name=self.analyzer_name,
            summary=". ".join(summary_parts) or "Screen analyzed with heuristic analyzer.",
            source=UIAwarenessSource.SCREENSHOT,
            regions=regions,
            elements=elements,
            anchors=anchors,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"window_title": window.title if window else None},
        )

    def build_awareness(
        self,
        request: UIAwarenessRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> UIAwarenessResult | None:
        analysis = self.analyze(
            VisionAnalysisRequest(
                capture=request.capture,
                image_bytes=request.image_bytes,
                image_path=request.image_path,
                analyzer_name=self.analyzer_name,
                include_ocr=request.include_ocr,
                include_ui_metadata=request.include_ui_tree,
                correlation_id=request.correlation_id,
                metadata=request.metadata,
            ),
            capture=capture,
            ocr_result=ocr_result,
            window=window,
        )
        return UIAwarenessResult(
            source=UIAwarenessSource.SCREENSHOT,
            window=window or (capture.window if capture else None),
            regions=analysis.regions,
            text_blocks=list(ocr_result.blocks) if ocr_result else [],
            elements=analysis.elements,
            anchors=analysis.anchors,
            summary=analysis.summary,
            source_confidence={"screenshot": 0.72, "ocr": 0.78 if ocr_result and ocr_result.blocks else 0.0},
            metadata={"analyzer_name": self.analyzer_name},
        )


class WindowsUIAwarenessBackend:
    analyzer_name = "windows_ui_awareness"

    def __init__(self, ui_adapter=None) -> None:
        self._ui_adapter = ui_adapter

    def health_check(self) -> dict[str, object]:
        return {"analyzer_name": self.analyzer_name, "healthy": self._ui_adapter is not None}

    def analyze(
        self,
        request: VisionAnalysisRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> VisionAnalysisResult:
        active = window or self._active_window()
        if active is None:
            raise VisionRuntimeError("ui metadata adapter is unavailable")
        region = ScreenRegion(
            left=active.rect.get("left", 0),
            top=active.rect.get("top", 0),
            right=active.rect.get("right", capture.width if capture else 0),
            bottom=active.rect.get("bottom", capture.height if capture else 0),
            label=active.title,
            confidence=0.95,
        )
        element = VisualElement(
            element_id=f"window-{active.handle}",
            kind=VisualElementKind.WINDOW,
            label=active.title,
            text=active.title,
            region=region,
            confidence=0.95,
            source=UIAwarenessSource.UI_TREE,
            actionable=False,
            metadata={"class_name": active.class_name, "process_id": active.process_id},
        )
        return VisionAnalysisResult(
            analyzer_name=self.analyzer_name,
            summary=f"Structured UI metadata available for {active.title}.",
            source=UIAwarenessSource.UI_TREE,
            regions=[region],
            elements=[element],
            anchors=[],
            metadata={"window_title": active.title},
        )

    def build_awareness(
        self,
        request: UIAwarenessRequest,
        *,
        capture: ScreenCaptureResult | None = None,
        ocr_result: OCRResult | None = None,
        window: WindowInfo | None = None,
    ) -> UIAwarenessResult | None:
        active = window or self._active_window()
        if active is None:
            return None
        rect = active.rect or {"left": 0, "top": 0, "right": capture.width if capture else 0, "bottom": capture.height if capture else 0}
        region = ScreenRegion(
            left=int(rect.get("left", 0)),
            top=int(rect.get("top", 0)),
            right=int(rect.get("right", capture.width if capture else 0)),
            bottom=int(rect.get("bottom", capture.height if capture else 0)),
            label=active.title,
            confidence=0.95,
        )
        element = VisualElement(
            element_id=f"window-{active.handle}",
            kind=VisualElementKind.WINDOW,
            label=active.title,
            text=active.title,
            region=region,
            confidence=0.95,
            source=UIAwarenessSource.UI_TREE,
            actionable=False,
            metadata={"class_name": active.class_name, "process_id": active.process_id},
        )
        return UIAwarenessResult(
            source=UIAwarenessSource.UI_TREE,
            window=active,
            regions=[region],
            text_blocks=list(ocr_result.blocks) if ocr_result else [],
            elements=[element],
            anchors=[],
            summary=f"UI metadata reports active window {active.title}.",
            source_confidence={"ui_tree": 0.95},
            metadata={"window_handle": active.handle},
        )

    def _active_window(self) -> WindowInfo | None:
        if self._ui_adapter is None or not hasattr(self._ui_adapter, "get_active_window"):
            return None
        return self._ui_adapter.get_active_window()


def _infer_kind(text: str) -> VisualElementKind:
    lowered = text.casefold()
    if "boton" in lowered or "button" in lowered or "guardar" in lowered:
        return VisualElementKind.BUTTON
    if "campo" in lowered or "input" in lowered or "buscar" in lowered:
        return VisualElementKind.INPUT
    if "menu" in lowered:
        return VisualElementKind.MENU
    return VisualElementKind.TEXT
