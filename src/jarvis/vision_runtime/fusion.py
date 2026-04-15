from __future__ import annotations

from .base import OCRResult, ScreenCaptureResult, UIAnchor, UIAwarenessRequest, UIAwarenessResult, UIAwarenessSource, VisualElement, WindowInfo


def fuse_awareness(
    request: UIAwarenessRequest,
    *,
    capture: ScreenCaptureResult | None,
    ocr_result: OCRResult | None,
    window: WindowInfo | None,
    awareness_results: list[UIAwarenessResult],
) -> UIAwarenessResult:
    if not awareness_results:
        return UIAwarenessResult(
            source=UIAwarenessSource.FUSED,
            window=window or (capture.window if capture else None),
            text_blocks=list(ocr_result.blocks) if ocr_result else [],
            summary="No structured UI awareness sources produced a result.",
            source_confidence={"ocr": 0.4 if ocr_result and ocr_result.blocks else 0.0},
            metadata={"degradation_policy": request.metadata.get("degradation_policy")},
        )

    base = awareness_results[0].model_copy(deep=True)
    base.source = UIAwarenessSource.FUSED if len(awareness_results) > 1 or ocr_result else awareness_results[0].source
    if base.window is None:
        base.window = window or (capture.window if capture else None)
    if ocr_result:
        base.text_blocks = _merge_text_blocks(base.text_blocks, ocr_result.blocks)
        if not base.summary:
            base.summary = ocr_result.text[:300]
        base.source_confidence.setdefault("ocr", 0.75 if ocr_result and ocr_result.blocks else 0.3)
    for candidate in awareness_results[1:]:
        base.regions = _dedupe_regions(base.regions + candidate.regions)
        base.text_blocks = _merge_text_blocks(base.text_blocks, candidate.text_blocks)
        base.elements = _merge_elements(base.elements, candidate.elements)
        base.anchors = _merge_anchors(base.anchors, candidate.anchors)
        base.fusion_warnings.extend(candidate.fusion_warnings)
        for name, confidence in candidate.source_confidence.items():
            base.source_confidence[name] = max(base.source_confidence.get(name, 0.0), confidence)
        if candidate.summary and candidate.summary not in base.summary:
            base.summary = "; ".join(part for part in [base.summary, candidate.summary] if part)
    base.regions = _dedupe_regions(base.regions)
    base.elements = _merge_elements([], base.elements)
    base.anchors = _merge_anchors([], base.anchors)
    base.fusion_warnings = list(dict.fromkeys(base.fusion_warnings))
    return base


def _merge_text_blocks(existing, incoming):
    merged = list(existing)
    seen = {(item.text, item.region.left, item.region.top, item.region.right, item.region.bottom) for item in existing}
    for item in incoming:
        key = (item.text, item.region.left, item.region.top, item.region.right, item.region.bottom)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _merge_elements(existing: list[VisualElement], incoming: list[VisualElement]) -> list[VisualElement]:
    merged = list(existing)
    seen = {(item.kind.value, item.label or item.text or "", item.region.left, item.region.top, item.region.right, item.region.bottom) for item in existing}
    for item in incoming:
        key = (item.kind.value, item.label or item.text or "", item.region.left, item.region.top, item.region.right, item.region.bottom)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _merge_anchors(existing: list[UIAnchor], incoming: list[UIAnchor]) -> list[UIAnchor]:
    merged = list(existing)
    seen = {(item.label, item.region.left, item.region.top, item.region.right, item.region.bottom) for item in existing}
    for item in incoming:
        key = (item.label, item.region.left, item.region.top, item.region.right, item.region.bottom)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _dedupe_regions(regions):
    merged = []
    seen = set()
    for item in regions:
        key = (item.left, item.top, item.right, item.bottom, item.label or "")
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged
