"""OpenCV camera + YuNet/SFace engine for the face guard."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import JarvisError

logger = logging.getLogger("jarvis.face_guard")


class OpenCVCamera:
    """Webcam via cv2.VideoCapture; released on close."""

    def __init__(self, *, camera_index: int = 0) -> None:
        import cv2

        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise JarvisError(
                "No pude abrir la camara. Revisa que no la este usando otra app.",
                component="face_guard",
                code="camera_unavailable",
                recoverable=True,
            )

    def read(self) -> tuple[bool, Any]:
        return self._capture.read()

    def release(self) -> None:
        try:
            self._capture.release()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


class OpenCVFaceEngine:
    """YuNet face detector + SFace embedder (opencv_zoo ONNX models)."""

    def __init__(self, settings: Settings) -> None:
        import cv2

        models_dir = Path(
            getattr(settings, "face_guard_models_dir", "models/face")
        )
        if not models_dir.is_absolute():
            models_dir = Path.cwd() / models_dir
        detector_path = models_dir / "yunet.onnx"
        recognizer_path = models_dir / "sface.onnx"
        if not detector_path.is_file() or not recognizer_path.is_file():
            raise JarvisError(
                f"Faltan los modelos de caras en {models_dir} "
                "(yunet.onnx / sface.onnx del opencv_zoo)",
                component="face_guard",
                code="face_guard_models_missing",
                recoverable=True,
            )
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(detector_path), "", (320, 320)
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(recognizer_path), ""
            )
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"No pude cargar los modelos de caras: {exc}",
                component="face_guard",
                code="face_guard_models_invalid",
                recoverable=True,
            ) from exc

    def detect_largest(self, frame: Any) -> Any | None:

        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        faces = self._detector.detect(frame)
        detections = faces[1] if faces and faces[1] is not None else []
        if len(detections) == 0:
            return None
        # largest box by area = closest person
        return max(detections, key=lambda f: float(f[2]) * float(f[3]))

    def embed(self, frame: Any, box: Any) -> list[float] | None:

        aligned = self._recognizer.alignCrop(frame, box)
        feature = self._recognizer.feature(aligned)
        return feature.flatten().tolist()
