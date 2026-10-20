"""OpenCV-based face detection, matching, and multi-frame identity consensus."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from data_policy import _safe_path, safe_file_path
from typing import Sequence

import cv2
import numpy as np

from config import DEFAULT_CONFIG
from logging_setup import get_logger


logger = get_logger("identity")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DetectedFace:
    raw: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class KnownIdentity:
    name: str
    feature: np.ndarray
    image_path: Path


@dataclass(frozen=True)
class IdentityMatch:
    name: str
    score: float | None
    matched: bool
    runner_up_score: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class IdentityDecision:
    name: str
    score: float | None
    confirmed: bool


@dataclass
class IdentityConsensus:
    """Require repeated matches before displaying an enrolled identity."""

    window_size: int = 5
    required_matches: int = 3
    ttl_frames: int = 90
    _observations: deque[tuple[str | None, float | None]] = field(
        default_factory=deque,
        init=False,
    )
    _confirmed_name: str | None = field(default=None, init=False)
    _confirmed_score: float | None = field(default=None, init=False)
    _last_confirmed_frame: int | None = field(default=None, init=False)
    _last_observation_frame: int | None = field(default=None, init=False)
    _suspended: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.window_size = max(1, self.window_size)
        self.required_matches = min(
            self.window_size,
            max(1, self.required_matches),
        )
        self.ttl_frames = max(1, self.ttl_frames)

    def observe(
        self,
        match: IdentityMatch | None,
        frame_number: int,
    ) -> IdentityDecision:
        # Expire cached evidence before a new observation can renew it.
        self.current(frame_number)
        if match is None:
            # No usable observation is not negative recognition evidence.
            return self.current(frame_number)
        if not match.matched:
            self.clear()
            return self.current(frame_number)
        previous_candidate = next((name for name, _ in reversed(self._observations) if name), None)
        if previous_candidate is not None and match.name != previous_candidate:
            # Neither confirmed nor pending votes may cross a contradictory observation.
            self.clear()
        accepted_name = match.name if match is not None and match.matched else None
        accepted_score = match.score if match is not None and accepted_name is not None else None
        if (
            accepted_name is not None
            and self._confirmed_name is not None
            and accepted_name != self._confirmed_name
        ):
            self._suspended = True
        elif accepted_name == self._confirmed_name and accepted_name is not None:
            self._suspended = False
        self._observations.append((accepted_name, accepted_score))
        self._last_observation_frame = frame_number
        while len(self._observations) > self.window_size:
            self._observations.popleft()

        counts = Counter(name for name, _ in self._observations if name is not None)
        if counts:
            candidate_name, candidate_count = counts.most_common(1)[0]
            candidate_scores = [
                score
                for name, score in self._observations
                if name == candidate_name and score is not None
            ]
            # Only fresh support for the candidate can renew confirmation.
            if candidate_count >= self.required_matches and accepted_name == candidate_name:
                self._confirmed_name = candidate_name
                self._confirmed_score = (
                    sum(candidate_scores) / len(candidate_scores) if candidate_scores else None
                )
                self._last_confirmed_frame = frame_number
                self._suspended = False
        return self.current(frame_number)

    def current(self, frame_number: int) -> IdentityDecision:
        if (
            self._last_observation_frame is not None
            and frame_number - self._last_observation_frame > self.ttl_frames
        ):
            self.clear()
        if (
            self._confirmed_name is not None
            and self._last_confirmed_frame is not None
            and frame_number - self._last_confirmed_frame > self.ttl_frames
        ):
            self.clear()

        return IdentityDecision(
            name=(self._confirmed_name if not self._suspended else None) or "Unknown",
            score=self._confirmed_score if not self._suspended else None,
            confirmed=self._confirmed_name is not None and not self._suspended,
        )

    def clear(self) -> None:
        self._observations.clear()
        self._confirmed_name = None
        self._confirmed_score = None
        self._last_confirmed_frame = None
        self._last_observation_frame = None
        self._suspended = False


class OpenCVFaceIdentifier:
    """Cross-platform face detection and matching using OpenCV DNN models."""

    def __init__(
        self,
        detector_model_path: Path = DEFAULT_CONFIG.face_detector_model_path,
        recognizer_model_path: Path = DEFAULT_CONFIG.face_recognizer_model_path,
        cosine_threshold: float = DEFAULT_CONFIG.identity_cosine_threshold,
        min_score_margin: float = DEFAULT_CONFIG.identity_min_score_margin,
        min_face_size: int = DEFAULT_CONFIG.identity_min_face_size,
        min_face_confidence: float = DEFAULT_CONFIG.identity_min_face_confidence,
    ) -> None:
        if not detector_model_path.exists():
            raise FileNotFoundError(f"Face detector model not found: {detector_model_path}")
        if not recognizer_model_path.exists():
            raise FileNotFoundError(f"Face recognizer model not found: {recognizer_model_path}")

        self.cosine_threshold = cosine_threshold
        self.min_score_margin = min_score_margin
        self.min_face_size = min_face_size
        self.min_face_confidence = min_face_confidence
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model_path),
            "",
            (320, 320),
            0.8,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model_path), "")

    def load_enrollments(self, enrollments_dir: Path) -> list[KnownIdentity]:
        enrollments_dir = _safe_path(enrollments_dir)
        if not enrollments_dir.exists():
            return []

        identities: list[KnownIdentity] = []
        for person_dir in sorted(path for path in enrollments_dir.iterdir() if path.is_dir()):
            person_dir = _safe_path(person_dir)
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = cv2.imread(str(safe_file_path(image_path)))
                if image is None:
                    logger.warning("Could not read enrollment image: %s", image_path)
                    continue

                face = self.detect_largest_face(image)
                if face is None or not self.is_face_usable(face):
                    logger.warning("Enrollment face did not pass quality checks: %s", image_path)
                    continue

                feature = self.extract_feature(image, face)
                if feature is None:
                    logger.warning("Could not encode enrollment face: %s", image_path)
                    continue

                identities.append(
                    KnownIdentity(
                        name=person_dir.name,
                        feature=feature,
                        image_path=image_path,
                    )
                )

        return identities

    def detect_largest_face(self, image: np.ndarray) -> DetectedFace | None:
        faces = self.detect_faces(image)
        return max(faces, key=lambda face: _face_area(face.box)) if faces else None

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        if image.size == 0:
            return []

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return []
        return [
            DetectedFace(
                raw=raw_face,
                box=face_box(raw_face, image_width=width, image_height=height),
                confidence=float(raw_face[14]),
            )
            for raw_face in faces
        ]

    def is_face_usable(self, face: DetectedFace) -> bool:
        x1, y1, x2, y2 = face.box
        return (
            face.confidence >= self.min_face_confidence
            and x2 - x1 >= self.min_face_size
            and y2 - y1 >= self.min_face_size
        )

    def extract_feature(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> np.ndarray | None:
        aligned = self.recognizer.alignCrop(image, face.raw)
        if aligned is None or aligned.size == 0:
            return None

        return self.recognizer.feature(aligned).copy()

    def identify(
        self,
        image: np.ndarray,
        face: DetectedFace,
        known_identities: Sequence[KnownIdentity],
    ) -> IdentityMatch:
        feature = self.extract_feature(image, face)
        if feature is None:
            return IdentityMatch(name="Unknown", score=None, matched=False)

        scores_by_name: dict[str, float] = {}
        for identity in known_identities:
            score = float(
                self.recognizer.match(
                    feature,
                    identity.feature,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            scores_by_name[identity.name] = max(
                score,
                scores_by_name.get(identity.name, -1.0),
            )

        if not scores_by_name:
            return IdentityMatch(name="Unknown", score=None, matched=False)

        ranked = sorted(scores_by_name.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else None
        margin = best_score - runner_up_score if runner_up_score is not None else best_score
        matched = best_score >= self.cosine_threshold and margin >= self.min_score_margin
        return IdentityMatch(
            name=best_name if matched else "Unknown",
            score=best_score,
            matched=matched,
            runner_up_score=runner_up_score,
            margin=margin,
        )


def face_box(
    raw_face: np.ndarray,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int, int, int]:
    x, y, width, height = raw_face[:4]
    x1 = max(0, int(round(float(x))))
    y1 = max(0, int(round(float(y))))
    x2 = max(x1, int(round(float(x + width))))
    y2 = max(y1, int(round(float(y + height))))
    if image_width is not None:
        x1 = min(x1, image_width)
        x2 = min(x2, image_width)
    if image_height is not None:
        y1 = min(y1, image_height)
        y2 = min(y2, image_height)
    return x1, y1, x2, y2


def offset_box(
    box: tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y


def _face_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)
