from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.safety import ensure_within_roots
from jarvis.tools.models import ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry
from jarvis.vision_runtime.base import (
    CaptureTargetType,
    ElementLocationRequest,
    OCRRequest,
    ScreenCaptureRequest,
    ScreenRegion,
    TextLocationRequest,
    UIAwarenessRequest,
    VisionAnalysisRequest,
)
from jarvis.vision_runtime.serialization import serialize_vision_operation_receipt
from jarvis.vision_runtime.service import VisionRuntimeService


class InspectImagePayload(BaseModel):
    image_path: str


class DescribeScreenPayload(BaseModel):
    capture: ScreenCaptureRequest | None = None


class VisionModule:
    name = "vision"
    description = "Screen capture, OCR, UI awareness and image inspection."

    def __init__(self, vision_runtime: VisionRuntimeService) -> None:
        self._vision = vision_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="vision.inspect_image", description="Inspect local image metadata.", payload_model=InspectImagePayload, handler=self._inspect_image, tags=("vision", "inspection")))
        registry.register(ActionDefinition(name="vision.capture_screen", description="Capture the screen or active window.", payload_model=ScreenCaptureRequest, handler=self._capture_screen, tags=("vision", "capture")))
        registry.register(ActionDefinition(name="vision.capture_window", description="Capture a named window.", payload_model=ScreenCaptureRequest, handler=self._capture_window, tags=("vision", "capture")))
        registry.register(ActionDefinition(name="vision.capture_region", description="Capture a region of the screen.", payload_model=ScreenCaptureRequest, handler=self._capture_region, tags=("vision", "capture")))
        registry.register(ActionDefinition(name="vision.extract_text", description="Extract visible text through OCR.", payload_model=OCRRequest, handler=self._extract_text, tags=("vision", "ocr")))
        registry.register(ActionDefinition(name="vision.describe_screen", description="Analyze the visible screen contents.", payload_model=VisionAnalysisRequest, handler=self._describe_screen, tags=("vision", "analysis")))
        registry.register(ActionDefinition(name="vision.locate_text", description="Locate text on screen and return matching regions.", payload_model=TextLocationRequest, handler=self._locate_text, tags=("vision", "grounding")))
        registry.register(ActionDefinition(name="vision.locate_element", description="Locate inferred UI elements on screen.", payload_model=ElementLocationRequest, handler=self._locate_element, tags=("vision", "grounding")))
        registry.register(ActionDefinition(name="vision.ui_awareness", description="Build a fused UI awareness model from screenshot, OCR and UI metadata.", payload_model=UIAwarenessRequest, handler=self._ui_awareness, tags=("vision", "awareness")))

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(ToolDefinition(name="screen.capture", description="Capture the screen, window or region.", input_model=ScreenCaptureRequest, handler=self._tool_capture, tags=("vision", "capture")))
        registry.register(ToolDefinition(name="screen.ocr", description="Extract text from a screenshot or provided image.", input_model=OCRRequest, handler=self._tool_ocr, tags=("vision", "ocr")))
        registry.register(ToolDefinition(name="screen.describe", description="Describe the current screen state.", input_model=VisionAnalysisRequest, handler=self._tool_describe, tags=("vision", "analysis")))
        registry.register(ToolDefinition(name="screen.locate_text", description="Locate text on the current screen.", input_model=TextLocationRequest, handler=self._tool_locate_text, tags=("vision", "grounding")))
        registry.register(ToolDefinition(name="screen.ui_awareness", description="Build a structured UI awareness result.", input_model=UIAwarenessRequest, handler=self._tool_ui_awareness, tags=("vision", "awareness")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="vision.inspect",
                module_name=self.name,
                intent="vision",
                description="Inspect images, capture screen content and run OCR or visual analysis.",
                action_names=("vision.inspect_image", "vision.capture_screen", "vision.extract_text", "vision.describe_screen"),
                tool_names=("image.inspect", "screen.capture", "screen.ocr", "screen.describe"),
                keywords=("imagen", "image", "foto", "ocr", "captura de pantalla", "screen capture", "analiza imagen"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
                supports_planning=False,
            ),
            plan_builder=self._build_vision_plan,
        )
        registry.register(
            CapabilityDescriptor(
                name="vision.screen_read",
                module_name=self.name,
                intent="screen_read",
                description="Read and describe visible on-screen information.",
                action_names=("vision.extract_text", "vision.describe_screen"),
                tool_names=("screen.ocr", "screen.describe"),
                keywords=("lee la pantalla", "screen read", "visible text", "ocr de pantalla"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
                supports_planning=False,
            ),
            plan_builder=self._build_screen_read_plan,
        )
        registry.register(
            CapabilityDescriptor(
                name="vision.ui_awareness",
                module_name=self.name,
                intent="ui_awareness",
                description="Build a structured awareness model of the current UI.",
                action_names=("vision.ui_awareness", "vision.locate_text", "vision.locate_element"),
                tool_names=("screen.ui_awareness", "screen.locate_text"),
                keywords=("ui awareness", "localiza boton en pantalla", "encuentra texto en pantalla", "find on screen"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
                supports_planning=False,
            ),
            plan_builder=self._build_ui_awareness_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _inspect_image(self, context: ActionContext, payload: InspectImagePayload) -> ActionResult:
        allowed_roots = (context.settings.resolved_workspace_root, *context.settings.resolved_research_roots)
        image_path = ensure_within_roots(payload.image_path, allowed_roots, "image inspection")
        with Image.open(image_path) as image:
            exif = image.getexif()
            return ActionResult(
                message="image inspected",
                data={
                    "path": str(Path(image_path)),
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "frames": getattr(image, "n_frames", 1),
                    "exif_tags": len(exif or {}),
                },
            )

    def _capture_screen(self, context: ActionContext, payload: ScreenCaptureRequest) -> ActionResult:
        receipt = self._vision.capture_screen(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _capture_window(self, context: ActionContext, payload: ScreenCaptureRequest) -> ActionResult:
        receipt = self._vision.capture_window(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _capture_region(self, context: ActionContext, payload: ScreenCaptureRequest) -> ActionResult:
        receipt = self._vision.capture_region(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _extract_text(self, context: ActionContext, payload: OCRRequest) -> ActionResult:
        receipt = self._vision.extract_text(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _describe_screen(self, context: ActionContext, payload: VisionAnalysisRequest) -> ActionResult:
        receipt = self._vision.analyze_image(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _locate_text(self, context: ActionContext, payload: TextLocationRequest) -> ActionResult:
        receipt = self._vision.locate_text(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _locate_element(self, context: ActionContext, payload: ElementLocationRequest) -> ActionResult:
        receipt = self._vision.locate_element(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _ui_awareness(self, context: ActionContext, payload: UIAwarenessRequest) -> ActionResult:
        receipt = self._vision.build_ui_awareness(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=serialize_vision_operation_receipt(receipt))

    def _tool_capture(self, context: ToolContext, payload: ScreenCaptureRequest) -> ToolResult:
        if payload.target_type in {CaptureTargetType.SCREEN, CaptureTargetType.ACTIVE_WINDOW}:
            receipt = self._vision.capture_screen(payload, correlation_id=context.correlation_id)
        elif payload.target_type == CaptureTargetType.WINDOW:
            receipt = self._vision.capture_window(payload, correlation_id=context.correlation_id)
        else:
            receipt = self._vision.capture_region(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_ocr(self, context: ToolContext, payload: OCRRequest) -> ToolResult:
        receipt = self._vision.extract_text(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_describe(self, context: ToolContext, payload: VisionAnalysisRequest) -> ToolResult:
        receipt = self._vision.analyze_image(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_locate_text(self, context: ToolContext, payload: TextLocationRequest) -> ToolResult:
        receipt = self._vision.locate_text(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_ui_awareness(self, context: ToolContext, payload: UIAwarenessRequest) -> ToolResult:
        receipt = self._vision.build_ui_awareness(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=serialize_vision_operation_receipt(receipt))

    @staticmethod
    def _build_vision_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if payload.get("image_path"):
            return [ActionStep(action="vision.inspect_image", payload=payload)]
        if payload.get("text"):
            payload.setdefault("awareness", {"capture": {"target_type": CaptureTargetType.ACTIVE_WINDOW.value}})
            return [ActionStep(action="vision.locate_text", payload=payload)]
        capture_payload = payload.get("capture", {})
        if capture_payload:
            capture_payload.setdefault("target_type", CaptureTargetType.ACTIVE_WINDOW.value)
            return [ActionStep(action="vision.describe_screen", payload={"capture": capture_payload})]
        return [ActionStep(action="vision.capture_screen", payload={"target_type": CaptureTargetType.SCREEN.value})]

    @staticmethod
    def _build_screen_read_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if payload.get("capture") is None:
            payload["capture"] = {"target_type": CaptureTargetType.ACTIVE_WINDOW.value}
        return [ActionStep(action="vision.describe_screen", payload=payload)]

    @staticmethod
    def _build_ui_awareness_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("capture", {"target_type": CaptureTargetType.ACTIVE_WINDOW.value})
        if payload.get("text"):
            return [ActionStep(action="vision.locate_text", payload=payload)]
        return [ActionStep(action="vision.ui_awareness", payload=payload)]
