from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEFAULT_CONFIG, AppConfig
from dependency_privacy import disable_ultralytics_telemetry
from model_manager import ensure_model, pose_model_spec


DEFAULT_MODEL_PATH = DEFAULT_CONFIG.pose_model_path
MIN_KEYPOINT_SCORE = 0.3
DEFAULT_SKELETON_COLOR = (0, 255, 0)


class MoveNetKeypoint(IntEnum):
    """COCO keypoint indexes shared by MoveNet and Ultralytics pose models."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


POSE_CONNECTIONS = (
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.RIGHT_SHOULDER),
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.LEFT_ELBOW),
    (MoveNetKeypoint.LEFT_ELBOW, MoveNetKeypoint.LEFT_WRIST),
    (MoveNetKeypoint.RIGHT_SHOULDER, MoveNetKeypoint.RIGHT_ELBOW),
    (MoveNetKeypoint.RIGHT_ELBOW, MoveNetKeypoint.RIGHT_WRIST),
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.LEFT_HIP),
    (MoveNetKeypoint.RIGHT_SHOULDER, MoveNetKeypoint.RIGHT_HIP),
    (MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.RIGHT_HIP),
    (MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.LEFT_KNEE),
    (MoveNetKeypoint.LEFT_KNEE, MoveNetKeypoint.LEFT_ANKLE),
    (MoveNetKeypoint.RIGHT_HIP, MoveNetKeypoint.RIGHT_KNEE),
    (MoveNetKeypoint.RIGHT_KNEE, MoveNetKeypoint.RIGHT_ANKLE),
    (MoveNetKeypoint.NOSE, MoveNetKeypoint.LEFT_EYE),
    (MoveNetKeypoint.NOSE, MoveNetKeypoint.RIGHT_EYE),
    (MoveNetKeypoint.LEFT_EYE, MoveNetKeypoint.LEFT_EAR),
    (MoveNetKeypoint.RIGHT_EYE, MoveNetKeypoint.RIGHT_EAR),
)


@dataclass(frozen=True)
class PoseResult:
    track_id: int | None
    box: tuple[int, int, int, int]
    landmarks: dict[int, tuple[float, float]]


class PoseAnalyzer:
    """Ultralytics multi-person pose tracking and skeleton rendering."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        disable_ultralytics_telemetry()
        from ultralytics import YOLO

        if model_path == config.pose_model_path:
            ensure_model(pose_model_spec(config), config)
        elif not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        self.confidence = config.pose_confidence
        self.iou = config.pose_iou
        self.min_keypoint_confidence = config.min_keypoint_confidence
        self.model = YOLO(str(model_path))

    def analyze(self, frame: np.ndarray) -> list[PoseResult]:
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            stream=False,
        )
        # Ultralytics versions annotate track() as either a list or an iterator.
        # Consume one frame without discarding the Results type information.
        result = next(iter(results), None)
        if result is None:
            return []
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None:
            return []

        height, width = frame.shape[:2]
        tracked_poses: list[PoseResult] = []
        for index in range(len(boxes)):
            track_id = None if boxes.id is None else int(boxes.id[index].item())
            landmarks = self.landmarks_from_keypoints(
                keypoints.xy[index],
                None if keypoints.conf is None else keypoints.conf[index],
                frame_width=width,
                frame_height=height,
                min_score=self.min_keypoint_confidence,
            )
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            raw_box = (int(x1), int(y1), int(x2), int(y2))
            box = _clamp_box(raw_box, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            tracked_poses.append(PoseResult(track_id, box, landmarks))

        return tracked_poses

    @staticmethod
    def landmarks_from_keypoints(
        keypoints: Any,
        confidence: Any,
        frame_width: int,
        frame_height: int,
        min_score: float = MIN_KEYPOINT_SCORE,
    ) -> dict[int, tuple[float, float]]:
        points = _as_numpy(keypoints)
        scores = (
            np.ones(len(points), dtype=np.float32) if confidence is None else _as_numpy(confidence)
        )
        landmarks: dict[int, tuple[float, float]] = {}
        for index, ((x, y), score) in enumerate(zip(points, scores)):
            if float(score) >= min_score:
                landmarks[index] = (
                    float(np.clip(x / frame_width, 0.0, 1.0)),
                    float(np.clip(y / frame_height, 0.0, 1.0)),
                )
        return landmarks

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: dict[int, tuple[float, float]],
        color: tuple[int, int, int] = DEFAULT_SKELETON_COLOR,
    ) -> None:
        height, width = frame.shape[:2]
        points = {
            index: (int(point[0] * width), int(point[1] * height))
            for index, point in landmarks.items()
        }

        for start, end in POSE_CONNECTIONS:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], color, 2)

        for point in points.values():
            cv2.circle(frame, point, 4, color, -1)

    def reset_tracking(self) -> None:
        """Clear ByteTrack state between unrelated live or recorded sources."""
        if hasattr(self.model, "predictor"):
            self.model.predictor = None


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clamp_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(frame_width, x1)),
        max(0, min(frame_height, y1)),
        max(0, min(frame_width, x2)),
        max(0, min(frame_height, y2)),
    )
