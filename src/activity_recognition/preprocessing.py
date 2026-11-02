"""Reproducible frame, pose, crop, and backbone-feature preprocessing."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from data_policy import ensure_private_directory, atomic_output, safe_file_path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable
from zipfile import BadZipFile

import cv2
import numpy as np

from activity_recognition.dataset import ActivitySample, _validate_manifest_samples
from activity_recognition.provenance import (
    array_hash,
    source_records,
    dataset_provenance,
    pose_extraction_provenance,
)
from activity_recognition.sampling import (
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    SAMPLING_VERSION,
    validate_sampling_interval,
)
from camera import CaptureClock
from config import AppConfig

if TYPE_CHECKING:
    import torch


KEYPOINT_COUNT = 17
DEFAULT_FRAMES_PER_SAMPLE = 16
DEFAULT_CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
S3D_MEAN = (0.43216, 0.394666, 0.37645)
S3D_STD = (0.22803, 0.22145, 0.216989)


def normalize_pose(
    landmarks: dict[int, tuple[float, float]],
    box: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    """Return 17 bounding-box-relative x/y/visibility keypoint triples."""
    features = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    if box is None:
        return features
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    for index, (x_normalized, y_normalized) in landmarks.items():
        if not 0 <= int(index) < KEYPOINT_COUNT:
            continue
        x_pixels = x_normalized * width
        y_pixels = y_normalized * height
        features[int(index)] = (
            float(np.clip((x_pixels - x1) / box_width, 0.0, 1.0)),
            float(np.clip((y_pixels - y1) / box_height, 0.0, 1.0)),
            1.0,
        )
    return features


def crop_person(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    output_size: int = DEFAULT_CROP_SIZE,
    padding_fraction: float = 0.08,
) -> np.ndarray:
    """Crop one person with padding, falling back to the full frame."""
    height, width = frame.shape[:2]
    if box is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * padding_fraction)
        pad_y = int((y2 - y1) * padding_fraction)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def sample_video_frames(
    video_path: Path,
    frame_count: int = DEFAULT_FRAMES_PER_SAMPLE,
    *,
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
) -> list[np.ndarray]:
    """Decode a fixed-duration source-time window with bounded frame storage."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    interval = validate_sampling_interval(sampling_interval_seconds)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        clock = CaptureClock(capture, recorded=True)
        origin = None
        last_frame = None
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = clock.timestamp()
            if origin is None:
                origin = timestamp
            elapsed = timestamp - origin
            while len(frames) < frame_count and len(frames) * interval < elapsed - 1e-8:
                frames.append(last_frame if last_frame is not None else frame)
            if len(frames) < frame_count and len(frames) * interval <= elapsed + 1e-8:
                frames.append(frame)
            last_frame = frame
        if last_frame is not None:
            frames.extend([last_frame] * (frame_count - len(frames)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return frames


def file_fingerprint(path: Path) -> str:
    """Hash source content without loading a video or NPZ entirely into memory."""
    digest = sha256()
    with safe_file_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_metadata(
    sample: ActivitySample, frames: int, interval: float, extraction: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": 3,
        "key": sample.key,
        "class": sample.hmdb51_class,
        "label_name": sample.label,
        "extraction": extraction,
        "source": str(Path(sample.video_path).resolve()),
        "source_sha256": file_fingerprint(Path(sample.video_path)),
        "frames_per_sample": frames,
        "sampling_interval_seconds": interval,
        "sampling_version": SAMPLING_VERSION,
        "crop_size": DEFAULT_CROP_SIZE,
        "pose_preprocessing": "coco17-box-relative-v1",
        "label": sample.label_index,
        "split": sample.split,
    }


def _valid_sample_cache(path: Path, expected: dict[str, Any]) -> bool:
    try:
        with np.load(safe_file_path(path), allow_pickle=False) as payload:
            if json.loads(str(payload["metadata"])) != expected:
                return False
            count = expected["frames_per_sample"]
            return (
                payload["pose"].shape == (count, KEYPOINT_COUNT, 3)
                and payload["crops"].shape == (count, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE, 3)
                and payload["pose"].dtype == np.float32
                and payload["crops"].dtype == np.uint8
                and bool(np.isfinite(payload["pose"]).all())
                and bool(((payload["pose"] >= 0) & (payload["pose"] <= 1)).all())
                and str(payload["pose_sha256"]) == array_hash(payload["pose"])
                and str(payload["crops_sha256"]) == array_hash(payload["crops"])
                and int(payload["label"]) == expected["label"]
                and str(payload["split"]) == expected["split"]
                and str(payload["source"]) == expected["source"]
            )
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return False


def validate_cached_samples(samples: Iterable[ActivitySample], metadata: dict[str, Any]) -> None:
    """Require manifest-bound artifacts and audit all partitions before experiments."""
    sample_list = list(samples)
    _validate_manifest_samples(sample_list)
    try:
        current = dataset_provenance(sample_list)
        registry = metadata.get("cache_provenance")
        extraction = metadata.get("extraction")
        if not isinstance(registry, dict) or not isinstance(extraction, dict):
            raise ValueError("missing provenance; regenerate legacy manifest/cache")
        if registry.keys() != current.keys():
            raise ValueError("partition membership differs from provenance registry")
        for sample in sample_list:
            if registry.get(sample.key) != current[sample.key]:
                raise ValueError(f"source/cache/partition provenance mismatch: {sample.key}")
            expected = _sample_metadata(
                sample,
                metadata["frames_per_sample"],
                validate_sampling_interval(metadata["sampling_interval_seconds"]),
                extraction,
            )
            if not _valid_sample_cache(Path(sample.cache_path), expected):
                raise ValueError(f"invalid arrays or extraction provenance: {sample.key}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid activity cache: {exc}. Run prepare_activity_data.py again."
        ) from exc


def cache_activity_samples(
    samples: Iterable[ActivitySample],
    *,
    frames_per_sample: int = DEFAULT_FRAMES_PER_SAMPLE,
    overwrite: bool = False,
    pose_analyzer: Any | None = None,
    extraction_provenance: dict[str, Any] | None = None,
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
) -> dict[str, int]:
    """Run YOLO pose once and cache normalized poses plus person crops."""
    sample_list = list(samples)
    _validate_manifest_samples(sample_list)
    source_records(sample_list)
    extraction = (
        extraction_provenance if extraction_provenance is not None else pose_extraction_provenance()
    )
    if frames_per_sample < 1:
        raise ValueError("frames_per_sample must be positive")
    interval = validate_sampling_interval(sampling_interval_seconds)
    metadata_by_key = {
        sample.key: _sample_metadata(sample, frames_per_sample, interval, extraction)
        for sample in sample_list
    }
    pending_samples = [
        sample
        for sample in sample_list
        if overwrite
        or not _valid_sample_cache(Path(sample.cache_path), metadata_by_key[sample.key])
    ]
    skipped = len(sample_list) - len(pending_samples)
    if not pending_samples:
        return {"completed": 0, "skipped": skipped, "failed": 0}

    if pose_analyzer is None:
        from pose import PoseAnalyzer

        config = AppConfig.from_env()
        pose_analyzer = PoseAnalyzer(config=config, model_path=config.pose_model_path)
    completed = failed = 0
    for position, sample in enumerate(pending_samples, start=1):
        cache_path = Path(sample.cache_path)
        ensure_private_directory(cache_path.parent)
        started = perf_counter()
        try:
            frames = sample_video_frames(
                Path(sample.video_path), frames_per_sample, sampling_interval_seconds=interval
            )
            reset = getattr(pose_analyzer, "reset_tracking", None)
            if callable(reset):
                reset()
            poses: list[np.ndarray] = []
            crops: list[np.ndarray] = []
            detected_count = 0
            for frame in frames:
                pose_results = pose_analyzer.analyze(frame)
                selected = max(
                    pose_results,
                    key=lambda result: _box_area(result.box),
                    default=None,
                )
                box = None if selected is None else selected.box
                landmarks = {} if selected is None else selected.landmarks
                detected_count += int(selected is not None)
                poses.append(normalize_pose(landmarks, box, frame.shape))
                crops.append(crop_person(frame, box))
            pose_array, crop_array = np.stack(poses), np.stack(crops)
            with atomic_output(cache_path) as temporary_path, temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    pose=pose_array,
                    crops=crop_array,
                    pose_sha256=np.array(array_hash(pose_array)),
                    crops_sha256=np.array(array_hash(crop_array)),
                    label=np.int64(sample.label_index),
                    split=np.array(sample.split),
                    source=np.array(str(Path(sample.video_path).resolve())),
                    metadata=np.array(json.dumps(metadata_by_key[sample.key], sort_keys=True)),
                    pose_detection_rate=np.float32(detected_count / len(frames)),
                    extraction_seconds=np.float64(perf_counter() - started),
                )
            completed += 1
            print(f"[{position}/{len(pending_samples)}] cached {Path(sample.video_path).name}")
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            failed += 1
            print(f"[{position}/{len(pending_samples)}] failed {sample.video_path}: {exc}")
    return {"completed": completed, "skipped": skipped, "failed": failed}


def load_cached_arrays(
    samples: Iterable[ActivitySample],
    field_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for sample in samples:
        cache_path = Path(sample.cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Activity cache missing: {cache_path}. Run prepare_activity_data.py."
            )
        with np.load(safe_file_path(cache_path), allow_pickle=False) as payload:
            values.append(np.asarray(payload[field_name]))
            labels.append(int(payload["label"]))
    return np.stack(values), np.asarray(labels, dtype=np.int64)


def image_tensor(crops: np.ndarray) -> "torch.Tensor":
    """Convert RGB uint8 crops to normalized MobileNet tensors."""
    import torch

    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def video_tensor(crops: np.ndarray) -> "torch.Tensor":
    """Convert RGB uint8 crops to one normalized S3D clip."""
    import torch

    tensor = torch.from_numpy(crops).permute(3, 0, 1, 2).float().div(255.0)
    mean = torch.tensor(S3D_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(S3D_STD).view(3, 1, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
