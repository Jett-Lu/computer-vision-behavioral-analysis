from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import shutil

from config import DEFAULT_CONFIG
from dependency_privacy import disable_onnx_telemetry
from data_policy import ensure_private_directory, atomic_output, safe_file_path

PROJECT_ROOT = DEFAULT_CONFIG.project_root
TMP_DIR = PROJECT_ROOT / ".tmp"
ensure_private_directory(Path(os.environ.get("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model_manager import EMOTION_MODEL_SHA256, file_sha256


DEFAULT_FACE_MODEL_PATH = DEFAULT_CONFIG.emotion_face_model_path
DEFAULT_CLASSIFIER_MODEL_PATH = DEFAULT_CONFIG.emotion_classifier_model_path
FACE_PADDING_RATIO = 0.12


@dataclass(frozen=True)
class FaceEmotionResult:
    box: tuple[int, int, int, int]
    keypoints: tuple[tuple[int, int], ...]
    label: str
    confidence: float
    uncertain: bool = False

    @property
    def display_label(self) -> str:
        return f"{self.label}?" if self.uncertain else self.label


@dataclass
class EmotionSmoother:
    """Stabilize per-person expression labels over a short rolling window."""

    window_seconds: float = 0.8
    uncertainty_threshold: float = 0.45
    _observations: deque[tuple[float, FaceEmotionResult]] = field(
        default_factory=deque,
        init=False,
    )

    def update(
        self,
        result: FaceEmotionResult | None,
        timestamp: float,
    ) -> FaceEmotionResult | None:
        if result is None:
            self._observations.clear()
            return None

        self._observations.append((timestamp, result))
        while self._observations and timestamp - self._observations[0][0] > self.window_seconds:
            self._observations.popleft()

        weights: defaultdict[str, float] = defaultdict(float)
        latest_index: dict[str, int] = {}
        for index, (_, observation) in enumerate(self._observations):
            weights[observation.label] += observation.confidence
            latest_index[observation.label] = index

        label = max(weights, key=lambda item: (weights[item], latest_index[item]))
        matching_confidences = [
            observation.confidence
            for _, observation in self._observations
            if observation.label == label
        ]
        confidence = sum(matching_confidences) / len(matching_confidences)
        return replace(
            result,
            label=label,
            confidence=confidence,
            uncertain=confidence < self.uncertainty_threshold,
        )


class FaceEmotionAnalyzer:
    """Detect a face within the pose box and classify its visible expression."""

    def __init__(
        self,
        model_path: Path = DEFAULT_FACE_MODEL_PATH,
        classifier_model_path: Path = DEFAULT_CLASSIFIER_MODEL_PATH,
        min_display_confidence: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face detector model not found: {model_path}")
        if not classifier_model_path.exists():
            raise FileNotFoundError(
                f"Expression classifier model not found: {classifier_model_path}"
            )

        _sync_emotiefflib_cache(classifier_model_path)
        disable_onnx_telemetry()
        self.recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name="enet_b2_8",
            device="cpu",
        )
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        detector_options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(detector_options)
        self.min_display_confidence = min_display_confidence

    def analyze(
        self,
        frame: np.ndarray,
        person_box: tuple[int, int, int, int] | None,
    ) -> FaceEmotionResult | None:
        if person_box is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        x1 = max(0, min(frame_width, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height, y1))
        y2 = max(0, min(frame_height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_height, crop_width = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        detection_result = self.detector.detect(image)
        if not detection_result.detections:
            return None

        face = max(
            detection_result.detections,
            key=lambda detection: detection.bounding_box.width * detection.bounding_box.height,
        )
        bbox = face.bounding_box
        fx1 = max(0, bbox.origin_x)
        fy1 = max(0, bbox.origin_y)
        fx2 = min(crop_width, bbox.origin_x + bbox.width)
        fy2 = min(crop_height, bbox.origin_y + bbox.height)
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        padded_box = self._padded_face_box((fx1, fy1, fx2, fy2), crop_width, crop_height)
        ex1, ey1, ex2, ey2 = padded_box
        face_rgb = crop_rgb[ey1:ey2, ex1:ex2]
        if face_rgb.size == 0:
            return None

        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        emotion_index = int(scores[0].argmax())
        label = self.recognizer.idx_to_emotion_class[emotion_index]
        confidence = float(scores[0][emotion_index])
        keypoints = tuple(
            (int(keypoint.x * crop_width) + x1, int(keypoint.y * crop_height) + y1)
            for keypoint in face.keypoints
        )

        return FaceEmotionResult(
            box=(fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1),
            keypoints=keypoints,
            label=label,
            confidence=confidence,
            uncertain=confidence < self.min_display_confidence,
        )

    def close(self) -> None:
        self.detector.close()

    @staticmethod
    def _padded_face_box(
        face_box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = face_box
        pad_x = int((x2 - x1) * FACE_PADDING_RATIO)
        pad_y = int((y2 - y1) * FACE_PADDING_RATIO)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )


def _sync_emotiefflib_cache(model_path: Path) -> None:
    """Place the verified model where EmotiEffLib expects to find it."""
    cache_path = safe_file_path(Path.home() / ".emotiefflib" / model_path.name)
    if cache_path.exists() and file_sha256(cache_path) == EMOTION_MODEL_SHA256:
        return

    with atomic_output(cache_path) as temporary_path:
        shutil.copyfile(model_path, temporary_path)
