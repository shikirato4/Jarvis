from __future__ import annotations

import hashlib
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, JarvisError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .awareness import VisionAnalyzerRegistry
from .base import (
    CaptureTargetType,
    ElementLocationMatch,
    ElementLocationRequest,
    OCRRequest,
    ScreenCaptureRequest,
    ScreenCaptureResult,
    TextLocationMatch,
    TextLocationRequest,
    UIAwarenessRequest,
    UIAwarenessResult,
    UIAwarenessSource,
    VisionAnalysisRequest,
    VisionOperationReceipt,
)
from .capture import ScreenCaptureBackendRegistry
from .fusion import fuse_awareness
from .ocr import OCRProviderRegistry, OCRService
from .safeguards import maybe_prepare_redaction, validate_capture_access, validate_capture_request, validate_capture_result, validate_ocr_request


class VisionRuntimeService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        event_bus: EventBus,
        capture_registry: ScreenCaptureBackendRegistry,
        ocr_registry: OCRProviderRegistry,
        awareness_registry: VisionAnalyzerRegistry,
        *,
        ui_metadata_adapter=None,
        logger: logging.Logger | None = None,
        resilience_controller=None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._event_bus = event_bus
        self._capture_registry = capture_registry
        self._ocr_registry = ocr_registry
        self._ocr = OCRService(settings, ocr_registry, logger=logger, resilience_controller=resilience_controller)
        self._awareness_registry = awareness_registry
        self._ui_metadata_adapter = ui_metadata_adapter
        self._logger = logger or logging.getLogger("jarvis.vision")
        self._operations = operation_registry
        self._ocr_cache: dict[str, tuple[float, VisionOperationReceipt]] = {}
        self._awareness_cache: dict[str, tuple[float, VisionOperationReceipt]] = {}

    def status(self) -> dict[str, object]:
        active_window = self._active_window()
        return {
            "capture_backends": [backend.health_check() for backend in self._capture_registry.list_backends()],
            "ocr_providers": self._ocr.health(),
            "awareness_backends": [analyzer.health_check() for analyzer in self._awareness_registry.list_analyzers()],
            "active_window": active_window.model_dump(mode="json") if active_window else None,
            "degradation_policy": self._settings.vision_degradation_policy,
            "capture_persistence_enabled": self._settings.vision_store_captures,
        }

    def capture_screen(self, request: ScreenCaptureRequest | None = None, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        payload = request or ScreenCaptureRequest(target_type=CaptureTargetType.SCREEN)
        return self._capture(payload, correlation_id=correlation_id, operation_name="vision.capture_screen")

    def capture_window(self, request: ScreenCaptureRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        return self._capture(request.model_copy(update={"target_type": CaptureTargetType.WINDOW}), correlation_id=correlation_id, operation_name="vision.capture_window")

    def capture_region(self, request: ScreenCaptureRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        return self._capture(request.model_copy(update={"target_type": CaptureTargetType.REGION}), correlation_id=correlation_id, operation_name="vision.capture_region")

    def extract_text(self, request: OCRRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        handle = self._begin_operation("vision.extract_text", correlation_id)
        try:
            validate_ocr_request(self._settings, request)
            capture_result = self._resolve_capture(request.capture, correlation_id=correlation_id) if request.capture else None
            cache_key = self._ocr_cache_key(request, capture_result)
            cached_receipt = self._cache_get(self._ocr_cache, cache_key)
            if cached_receipt is not None:
                return cached_receipt.model_copy(update={"correlation_id": correlation_id})
            ocr_result = self._ocr.extract_text(request, capture=capture_result)
            if len(ocr_result.blocks) > self._settings.vision_ocr_max_blocks:
                raise CapabilityUnavailableError("ocr block limit exceeded", component="vision_runtime")
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"block_count": len(ocr_result.blocks)})
            receipt = self._publish_success(
                correlation_id=correlation_id,
                operation_name="vision.extract_text",
                started_at=started_at,
                provider=ocr_result.provider_name,
                latency_ms=ocr_result.latency_ms,
                capture_target=capture_result.target_type.value if capture_result else None,
                fallback_used=ocr_result.fallback_used,
                capture_result=capture_result,
                ocr_result=ocr_result,
                data={
                    "text": ocr_result.text,
                    "block_count": len(ocr_result.blocks),
                    "language": ocr_result.language,
                    "ocr_ms": ocr_result.latency_ms,
                },
            )
            self._cache_put(self._ocr_cache, cache_key, receipt, ttl_seconds=1.0)
            return receipt
        except Exception as exc:  # noqa: BLE001
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc))
            self._publish_failure(correlation_id, "vision.extract_text", started_at, error=exc)
            raise

    def analyze_image(self, request: VisionAnalysisRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            capture_result = self._resolve_analysis_capture(request, correlation_id=correlation_id)
            ocr_result = None
            if request.include_ocr:
                ocr_result = self._ocr.extract_text(
                    OCRRequest(
                        capture=request.capture,
                        image_bytes=request.image_bytes,
                        image_path=request.image_path,
                        language=self._settings.vision_default_ocr_language,
                        correlation_id=correlation_id,
                    ),
                    capture=capture_result,
                )
            analyzer = self._resolve_analyzer(request.analyzer_name)
            analysis = analyzer.analyze(request, capture=capture_result, ocr_result=ocr_result, window=self._active_window())
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name="vision.analyze_image",
                started_at=started_at,
                backend=capture_result.backend_name if capture_result else None,
                provider=ocr_result.provider_name if ocr_result else None,
                analyzer=analysis.analyzer_name,
                latency_ms=analysis.latency_ms,
                capture_target=capture_result.target_type.value if capture_result else None,
                fallback_used=ocr_result.fallback_used if ocr_result else False,
                capture_result=capture_result,
                ocr_result=ocr_result,
                analysis_result=analysis,
                data={"summary": analysis.summary, "element_count": len(analysis.elements)},
            )
        except Exception as exc:  # noqa: BLE001
            self._publish_failure(correlation_id, "vision.analyze_image", started_at, error=exc)
            raise

    def describe_active_window(self, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        return self.build_ui_awareness(
            UIAwarenessRequest(
                capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW),
                include_ocr=True,
                include_ui_tree=True,
                correlation_id=correlation_id,
            ),
            correlation_id=correlation_id,
            operation_name="vision.describe_active_window",
        )

    def locate_text(self, request: TextLocationRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            awareness_receipt = self.build_ui_awareness(
                request.awareness or UIAwarenessRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW), correlation_id=correlation_id),
                correlation_id=correlation_id,
                operation_name="vision.locate_text.awareness",
            )
            awareness = awareness_receipt.awareness_result or UIAwarenessResult()
            needle = request.text if request.case_sensitive else request.text.casefold()
            matches = [
                TextLocationMatch(
                    text=block.text,
                    region=block.region,
                    confidence=block.confidence or 0.0,
                    source=UIAwarenessSource.OCR,
                    metadata={"awareness_summary": awareness.summary},
                )
                for block in awareness.text_blocks
                if needle in (block.text if request.case_sensitive else block.text.casefold())
            ]
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name="vision.locate_text",
                started_at=started_at,
                capture_target=awareness_receipt.capture_target,
                awareness_result=awareness,
                data={"matches": [item.model_dump(mode="json") for item in matches], "count": len(matches)},
            )
        except Exception as exc:  # noqa: BLE001
            self._publish_failure(correlation_id, "vision.locate_text", started_at, error=exc)
            raise

    def locate_element(self, request: ElementLocationRequest, *, correlation_id: str | None = None) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            awareness_receipt = self.build_ui_awareness(
                request.awareness or UIAwarenessRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW), correlation_id=correlation_id),
                correlation_id=correlation_id,
                operation_name="vision.locate_element.awareness",
            )
            awareness = awareness_receipt.awareness_result or UIAwarenessResult()
            matches = []
            for element in awareness.elements:
                if request.kind is not None and element.kind != request.kind:
                    continue
                if request.label is not None:
                    candidate = (element.label or element.text or "").casefold()
                    if request.label.casefold() not in candidate:
                        continue
                matches.append(
                    ElementLocationMatch(
                        element_id=element.element_id,
                        kind=element.kind,
                        label=element.label,
                        region=element.region,
                        confidence=element.confidence,
                        source=element.source,
                        metadata=element.metadata,
                    )
                )
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name="vision.locate_element",
                started_at=started_at,
                capture_target=awareness_receipt.capture_target,
                awareness_result=awareness,
                data={"matches": [item.model_dump(mode="json") for item in matches], "count": len(matches)},
            )
        except Exception as exc:  # noqa: BLE001
            self._publish_failure(correlation_id, "vision.locate_element", started_at, error=exc)
            raise

    def build_ui_awareness(
        self,
        request: UIAwarenessRequest,
        *,
        correlation_id: str | None = None,
        operation_name: str = "vision.ui_awareness",
    ) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            capture_result = self._resolve_awareness_capture(request, correlation_id=correlation_id)
            cache_key = self._awareness_cache_key(request, capture_result)
            cached_receipt = self._cache_get(self._awareness_cache, cache_key)
            if cached_receipt is not None:
                return cached_receipt.model_copy(update={"correlation_id": correlation_id})
            ocr_result = None
            ocr_started = time.perf_counter()
            if request.include_ocr:
                ocr_result = self._ocr.extract_text(
                    OCRRequest(
                        provider_name=request.ocr_provider_name,
                        capture=request.capture,
                        image_bytes=request.image_bytes,
                        image_path=request.image_path,
                        language=self._settings.vision_default_ocr_language,
                        correlation_id=correlation_id,
                        metadata=request.metadata,
                    ),
                    capture=capture_result,
                )
            ocr_ms = (time.perf_counter() - ocr_started) * 1000 if request.include_ocr else 0.0
            awareness_started = time.perf_counter()
            awareness_results = []
            if request.include_ui_tree:
                for analyzer in self._awareness_registry.list_analyzers():
                    result = analyzer.build_awareness(request, capture=capture_result, ocr_result=ocr_result, window=self._active_window())
                    if result is not None:
                        awareness_results.append(result)
            awareness = fuse_awareness(
                request,
                capture=capture_result,
                ocr_result=ocr_result,
                window=self._active_window(),
                awareness_results=awareness_results,
            )
            awareness_ms = (time.perf_counter() - awareness_started) * 1000
            receipt = self._publish_success(
                correlation_id=correlation_id,
                operation_name=operation_name,
                started_at=started_at,
                backend=capture_result.backend_name if capture_result else None,
                provider=ocr_result.provider_name if ocr_result else None,
                analyzer=request.analyzer_name or self._settings.vision_awareness_backend_default,
                capture_target=capture_result.target_type.value if capture_result else request.capture.target_type.value if request.capture else None,
                fallback_used=ocr_result.fallback_used if ocr_result else False,
                capture_result=capture_result,
                ocr_result=ocr_result,
                awareness_result=awareness,
                data={
                    "summary": awareness.summary,
                    "element_count": len(awareness.elements),
                    "anchor_count": len(awareness.anchors),
                    "ocr_ms": round(ocr_ms, 2),
                    "awareness_ms": round(awareness_ms, 2),
                },
            )
            self._cache_put(self._awareness_cache, cache_key, receipt, ttl_seconds=0.5)
            return receipt
        except Exception as exc:  # noqa: BLE001
            self._publish_failure(correlation_id, operation_name, started_at, error=exc)
            raise

    def _capture(self, request: ScreenCaptureRequest, *, correlation_id: str | None, operation_name: str) -> VisionOperationReceipt:
        correlation_id = correlation_id or request.correlation_id or str(uuid4())
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        try:
            result = self._resolve_capture(request, correlation_id=correlation_id)
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name=operation_name,
                started_at=started_at,
                backend=result.backend_name,
                latency_ms=round((time.perf_counter() - started_perf) * 1000, 2),
                capture_target=result.target_type.value,
                capture_result=result,
                data={"width": result.width, "height": result.height, "capture_ms": round((time.perf_counter() - started_perf) * 1000, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            self._publish_failure(correlation_id, operation_name, started_at, error=exc)
            raise

    def _resolve_capture(self, request: ScreenCaptureRequest, *, correlation_id: str) -> ScreenCaptureResult:
        validate_capture_access(self._mode_manager)
        validate_capture_request(self._settings, request)
        candidates = self._candidate_capture_backends(request)
        if not candidates:
            requested = request.backend_name or self._settings.vision_capture_backend_default
            raise CapabilityUnavailableError(f"capture backend '{requested}' is not registered", component="vision_runtime")
        last_error: Exception | None = None
        for backend_name in candidates:
            backend = self._capture_registry.get(backend_name)
            if backend is None:
                continue
            self._logger.info(
                "vision_capture_attempt",
                extra={
                    "correlation_id": correlation_id,
                    "backend_usado": backend_name,
                    "target_type": request.target_type.value,
                },
            )
            try:
                result = backend.capture(request.model_copy(update={"correlation_id": correlation_id, "backend_name": backend_name}))
                result = maybe_prepare_redaction(self._settings, result)
                validate_capture_result(self._settings, result)
                if request.persist:
                    result = self._persist_capture(result, correlation_id=correlation_id)
                self._logger.info(
                    "vision_capture_success",
                    extra={
                        "correlation_id": correlation_id,
                        "backend_usado": backend_name,
                        "target_type": request.target_type.value,
                        "width": result.width,
                        "height": result.height,
                    },
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.warning(
                    "vision_capture_failed",
                    extra={
                        "correlation_id": correlation_id,
                        "backend_usado": backend_name,
                        "target_type": request.target_type.value,
                        "error": str(exc),
                    },
                )
        assert last_error is not None
        raise JarvisError(str(last_error), component="vision_runtime", code="capture_failed", recoverable=True)

    def _resolve_analysis_capture(self, request: VisionAnalysisRequest, *, correlation_id: str) -> ScreenCaptureResult | None:
        if request.capture is not None:
            return self._resolve_capture(request.capture, correlation_id=correlation_id)
        return self._load_capture_from_request(request.image_bytes, request.image_path)

    def _resolve_awareness_capture(self, request: UIAwarenessRequest, *, correlation_id: str) -> ScreenCaptureResult | None:
        if request.capture is not None:
            return self._resolve_capture(request.capture, correlation_id=correlation_id)
        return self._load_capture_from_request(request.image_bytes, request.image_path)

    def _load_capture_from_request(self, image_bytes: bytes | None, image_path: str | None) -> ScreenCaptureResult | None:
        if image_bytes is None and image_path is None:
            return None
        if image_bytes is not None:
            if len(image_bytes) > self._settings.vision_max_image_bytes:
                raise CapabilityUnavailableError("provided image exceeds maximum byte budget", component="vision_runtime")
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.width, image.height
            return ScreenCaptureResult(backend_name="provided_image", target_type=CaptureTargetType.SCREEN, width=width, height=height, image_bytes=image_bytes)
        assert image_path is not None
        if Path(image_path).stat().st_size > self._settings.vision_max_image_bytes:
            raise CapabilityUnavailableError("provided image exceeds maximum byte budget", component="vision_runtime")
        with Image.open(image_path) as image:
            width, height = image.width, image.height
        return ScreenCaptureResult(backend_name="provided_image", target_type=CaptureTargetType.SCREEN, width=width, height=height, image_path=image_path)

    def _resolve_analyzer(self, analyzer_name: str | None):
        name = analyzer_name or self._settings.vision_awareness_backend_default
        analyzer = self._awareness_registry.get(name)
        if analyzer is None:
            raise CapabilityUnavailableError(f"vision analyzer '{name}' is not registered", component="vision_runtime")
        return analyzer

    def _persist_capture(self, result: ScreenCaptureResult, *, correlation_id: str) -> ScreenCaptureResult:
        if result.image_bytes is None:
            return result
        capture_dir = self._settings.resolved_vision_capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        file_path = capture_dir / f"{correlation_id}.{result.image_format.casefold()}"
        file_path.write_bytes(result.image_bytes)
        metadata = dict(result.metadata)
        metadata["persisted"] = True
        return result.model_copy(update={"image_path": str(file_path), "metadata": metadata})

    def _active_window(self):
        if self._ui_metadata_adapter is None or not hasattr(self._ui_metadata_adapter, "get_active_window"):
            return None
        return self._ui_metadata_adapter.get_active_window()

    def _publish_success(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        started_at: datetime,
        data: dict[str, object],
        backend: str | None = None,
        provider: str | None = None,
        analyzer: str | None = None,
        latency_ms: float | None = None,
        capture_target: str | None = None,
        fallback_used: bool = False,
        capture_result=None,
        ocr_result=None,
        analysis_result=None,
        awareness_result=None,
    ) -> VisionOperationReceipt:
        receipt = VisionOperationReceipt(
            correlation_id=correlation_id,
            operation_name=operation_name,
            success=True,
            message=operation_name,
            backend=backend,
            provider=provider,
            analyzer=analyzer,
            latency_ms=latency_ms,
            capture_target=capture_target,
            fallback_used=fallback_used,
            capture_result=capture_result,
            ocr_result=ocr_result,
            analysis_result=analysis_result,
            awareness_result=awareness_result,
            data=data,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish(
            "vision.executed",
            {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "backend": backend,
                "provider": provider,
                "analyzer": analyzer,
                "latency_ms": latency_ms,
                "capture_target": capture_target,
                "fallback_used": fallback_used,
                "data": data,
            },
        )
        return receipt

    def _publish_failure(self, correlation_id: str, operation_name: str, started_at: datetime, *, error: Exception) -> None:
        elapsed_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        self._logger.exception("vision_operation_failed", extra={"operation_name": operation_name, "correlation_id": correlation_id})
        self._event_bus.publish(
            "vision.failed",
            {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "latency_ms": elapsed_ms,
                "error": str(error),
            },
        )

    def _begin_operation(self, operation_name: str, correlation_id: str):
        if self._operations is None:
            return None
        return self._operations.begin(
            service_name="vision_runtime",
            operation_name=operation_name,
            correlation_id=correlation_id,
            timeout_ms=self._settings.vision_watchdog_timeout_ms,
            watchdog_timeout_ms=self._settings.vision_watchdog_timeout_ms,
        )

    def _candidate_capture_backends(self, request: ScreenCaptureRequest) -> list[str]:
        names: list[str] = []
        preferred = (
            request.backend_name,
            self._settings.vision_capture_backend_default,
        )
        for name in preferred:
            if name and name not in names:
                names.append(name)
        if self._settings.ui_backend_kind == "in_memory":
            if "in_memory_screen" not in names:
                names.append("in_memory_screen")
            return names
        for name in ("windows_mss", "windows_screen"):
            if name not in names:
                names.append(name)
        for backend in self._capture_registry.list_backends():
            if backend.backend_name == "in_memory_screen":
                continue
            if backend.backend_name not in names:
                names.append(backend.backend_name)
        return names

    def _ocr_cache_key(self, request: OCRRequest, capture_result: ScreenCaptureResult | None) -> str | None:
        if capture_result is None and request.image_bytes is None and request.image_path is None:
            return None
        image_hash = self._image_hash(capture_result.image_bytes if capture_result else request.image_bytes, capture_result.image_path if capture_result else request.image_path)
        if image_hash is None:
            return None
        return f"ocr:{request.provider_name or 'default'}:{request.language or self._settings.vision_default_ocr_language}:{image_hash}"

    def _awareness_cache_key(self, request: UIAwarenessRequest, capture_result: ScreenCaptureResult | None) -> str | None:
        if capture_result is None and request.image_bytes is None and request.image_path is None:
            return None
        image_hash = self._image_hash(capture_result.image_bytes if capture_result else request.image_bytes, capture_result.image_path if capture_result else request.image_path)
        if image_hash is None:
            return None
        return f"aware:{request.include_ocr}:{request.include_ui_tree}:{request.ocr_provider_name or 'default'}:{request.analyzer_name or self._settings.vision_awareness_backend_default}:{image_hash}"

    @staticmethod
    def _image_hash(image_bytes: bytes | None, image_path: str | None) -> str | None:
        if image_bytes:
            return hashlib.sha1(image_bytes).hexdigest()
        if image_path:
            path = Path(image_path)
            if path.exists():
                stat = path.stat()
                return hashlib.sha1(f"{path.resolve(strict=False)}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()
        return None

    @staticmethod
    def _cache_get(cache: dict[str, tuple[float, VisionOperationReceipt]], key: str | None) -> VisionOperationReceipt | None:
        if not key:
            return None
        cached = cache.get(key)
        if cached is None:
            return None
        expires_at, receipt = cached
        if expires_at < time.perf_counter():
            cache.pop(key, None)
            return None
        return receipt

    @staticmethod
    def _cache_put(cache: dict[str, tuple[float, VisionOperationReceipt]], key: str | None, receipt: VisionOperationReceipt, *, ttl_seconds: float) -> None:
        if not key:
            return
        cache[key] = (time.perf_counter() + ttl_seconds, receipt)
