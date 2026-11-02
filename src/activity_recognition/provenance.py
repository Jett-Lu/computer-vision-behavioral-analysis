"""Offline experiment evidence and random-state isolation; no live inference changes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import random
import inspect
from functools import wraps
from typing import Any, Iterable, Callable, ParamSpec, TypeVar
from urllib.parse import urlsplit

import numpy as np
import torch

from data_policy import safe_file_path, ensure_private_directory


def file_hash(path: Path) -> str:
    digest = sha256()
    with safe_file_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "torch", "torchvision", "ultralytics", "opencv-python"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "versions": versions(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def source_records(samples: Iterable[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    paths: set[str] = set()
    caches: set[str] = set()
    for sample in samples:
        path = safe_file_path(Path(sample.video_path))
        source = str(path.resolve())
        cache = str(safe_file_path(Path(sample.cache_path)).resolve())
        if sample.key in result or source in paths or cache in caches:
            raise ValueError("Duplicate activity source/key/cache")
        digest = file_hash(path)
        if digest in hashes and hashes[digest] != sample.split:
            raise ValueError("Duplicate source video content across data partitions")
        hashes[digest] = sample.split
        paths.add(source)
        caches.add(cache)
        result[sample.key] = {**asdict(sample), "source_sha256": digest}
    return result


def dataset_provenance(samples: Iterable[Any]) -> dict[str, dict[str, Any]]:
    records = source_records(samples)
    for item in records.values():
        item["cache_sha256"] = file_hash(Path(item["cache_path"]))
    return records


def annotation_provenance(paths: Iterable[Path], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "files": [
            {"path": str(safe_file_path(path).resolve()), "sha256": file_hash(path)}
            for path in paths
        ],
    }


def cache_quality_report(samples: Iterable[Any]) -> dict[str, Any]:
    """Describe visibility and imbalance without claiming ground-truth pose quality."""
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples:
        counts = groups.setdefault(sample.split, {}).setdefault(
            sample.label,
            {
                "samples": 0,
                "frames": 0,
                "missing_keypoints": 0,
                "keypoints": 0,
                "no_visible_pose_frames": 0,
            },
        )
        with np.load(safe_file_path(Path(sample.cache_path)), allow_pickle=False) as data:
            visible = data["pose"][:, :, 2] > 0
        counts["samples"] += 1
        counts["frames"] += len(visible)
        counts["keypoints"] += visible.size
        counts["missing_keypoints"] += int((~visible).sum())
        counts["no_visible_pose_frames"] += int((~visible.any(axis=1)).sum())
    for classes in groups.values():
        for counts in classes.values():
            counts["missing_keypoint_fraction"] = counts["missing_keypoints"] / counts["keypoints"]
            counts["no_visible_pose_frame_fraction"] = (
                counts["no_visible_pose_frames"] / counts["frames"]
            )
    return {
        "by_partition": groups,
        "scope": "cached keypoint visibility; not pose accuracy or identity continuity",
    }


def pose_extraction_provenance() -> dict[str, Any]:
    from config import AppConfig
    from pose import TRACKER_CONFIG_PATH

    config = AppConfig.from_env()
    root = Path(__file__).resolve().parent
    return {
        "version": 3,
        "weights_sha256": file_hash(config.pose_model_path),
        "pose_confidence": config.pose_confidence,
        "pose_detection_confidence": config.pose_detection_confidence,
        "pose_iou": config.pose_iou,
        "min_keypoint_confidence": config.min_keypoint_confidence,
        "tracker_sha256": file_hash(TRACKER_CONFIG_PATH),
        "implementation": {
            name: file_hash(root / name) for name in ("preprocessing.py", "sampling.py")
        },
        "pose_implementation": file_hash(root.parent / "pose.py"),
        "clock_implementation": file_hash(root.parent / "camera.py"),
        "dependencies": versions(),
        "person_selection": "largest-box",
        "crop_padding": 0.08,
        "crop_size": 224,
        "crop_color": "RGB",
        "resize": "INTER_AREA",
        "missing_pose": "zero-triples",
        "missing_person_crop": "full-frame",
    }


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def model_seed(seed: int, model_name: str) -> int:
    return int.from_bytes(sha256(f"{seed}:{model_name}".encode()).digest()[:4], "big")


def backbone_weight_fingerprint(model_name: str) -> str:
    from torchvision.models import MobileNet_V2_Weights
    from torchvision.models.video import S3D_Weights

    weights = (
        MobileNet_V2_Weights.IMAGENET1K_V2 if model_name == "cnn" else S3D_Weights.KINETICS400_V1
    )
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlsplit(weights.url).path).name
    return file_hash(path)


@contextmanager
def isolated_rng(seed: int):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(seed)
            np.random.seed(seed)
            # Seed only generators whose state this context preserves.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            torch.cuda.manual_seed_all(seed)
            if mps_state is not None:
                torch.mps.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def require_fresh_outputs(directory: Path, names: Iterable[str]) -> None:
    ensure_private_directory(directory)
    for name in names:
        if safe_file_path(directory / name).exists():
            raise FileExistsError(
                f"Preserving existing experiment artifact: {directory / name}. Use a new run directory."
            )


P = ParamSpec("P")
R = TypeVar("R")


def reproducible_training(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        arguments = inspect.signature(function).bind(*args, **kwargs).arguments
        with isolated_rng(model_seed(arguments["seed"], arguments["model_name"])):
            return function(*args, **kwargs)

    return wrapped
