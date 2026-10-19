"""Evaluate trained activity heads on the untouched HMDB51 test split."""

from __future__ import annotations

import platform
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from data_policy import ensure_private_directory
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

from activity_recognition.dataset import (
    ActivitySample,
    load_manifest,
    samples_for_split,
)
from activity_recognition.labels import ACTIVITY_LABELS, LABEL_TO_INDEX
from activity_recognition.metrics import (
    classification_metrics,
    confusion_matrix,
    save_confusion_matrix,
    write_comparison_csv,
    write_metrics_json,
    write_predictions_csv,
)
from activity_recognition.models import (
    MODEL_NAMES,
    build_activity_classifier,
    build_mobilenet_extractor,
    build_s3d_extractor,
)
from activity_recognition.preprocessing import image_tensor, validate_cached_samples, video_tensor
from activity_recognition.train import (
    cache_backbone_features,
    load_model_features,
    resolve_device,
    set_reproducible_seed,
)


def evaluate_all_models(
    manifest_path: Path,
    feature_root: Path,
    model_root: Path,
    results_root: Path,
    *,
    device_name: str = "auto",
    latency_samples: int = 20,
    overwrite_features: bool = False,
) -> dict[str, dict[str, Any]]:
    if latency_samples < 1:
        raise ValueError("latency_samples must be positive")
    metadata, samples = load_manifest(manifest_path)
    test_samples = samples_for_split(samples, "test")
    if not test_samples:
        raise ValueError("Manifest contains no official test samples")
    validate_cached_samples(test_samples, metadata)
    device = resolve_device(device_name)
    seed = int(metadata.get("seed", 2026))
    set_reproducible_seed(seed)
    ensure_private_directory(results_root)
    results: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    prediction_rows: list[dict] = []

    for model_name in MODEL_NAMES:
        checkpoint_path = model_root / f"{model_name}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Activity checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        validate_checkpoint_labels(checkpoint, checkpoint_path)
        if checkpoint.get("model_name") != model_name:
            raise ValueError(f"Checkpoint model name does not match filename: {checkpoint_path}")
        if checkpoint.get("frames_per_sample") != metadata.get("frames_per_sample"):
            raise ValueError(f"Checkpoint frame count does not match manifest: {checkpoint_path}")
        if checkpoint.get("sampling_interval_seconds") != metadata.get("sampling_interval_seconds"):
            raise ValueError(
                f"Checkpoint sampling interval does not match manifest: {checkpoint_path}"
            )
        if model_name in {"cnn", "advanced"}:
            cache_backbone_features(
                test_samples,
                feature_root,
                model_name,
                device,
                overwrite=overwrite_features,
            )
        model = build_activity_classifier(
            model_name,
            int(checkpoint["input_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        features, expected = load_model_features(
            test_samples,
            model_name,
            feature_root,
        )
        normalized = _standardize(features, checkpoint)
        with torch.inference_mode():
            logits = model(torch.from_numpy(normalized).float().to(device))
            predicted = logits.argmax(dim=1).cpu().numpy()
        matrix = confusion_matrix(expected, predicted, len(ACTIVITY_LABELS))
        metrics: dict[str, Any] = dict(classification_metrics(matrix))
        metrics["averaging"] = "macro"
        metrics["training_time_seconds"] = float(checkpoint["training_time_seconds"])
        metrics["inference_latency_ms"] = measure_inference_latency(
            model_name,
            model,
            checkpoint,
            test_samples[:latency_samples],
            device,
        )
        metrics["latency_scope"] = (
            "classifier inference from cached YOLO pose features; excludes YOLO"
            if model_name == "mlp"
            else (
                "cached-crop tensor preprocessing, frozen MobileNetV2 inference, "
                "and classifier inference"
                if model_name == "cnn"
                else (
                    "cached-clip tensor preprocessing, frozen S3D inference, "
                    "and classifier inference"
                )
            )
        )
        results[model_name] = metrics
        for sample, expected_index, predicted_index in zip(
            test_samples,
            expected,
            predicted,
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "sample_key": sample.key,
                    "source_video": Path(sample.video_path).name,
                    "split": sample.split,
                    "expected_index": int(expected_index),
                    "expected_label": ACTIVITY_LABELS[int(expected_index)],
                    "predicted_index": int(predicted_index),
                    "predicted_label": ACTIVITY_LABELS[int(predicted_index)],
                }
            )
        checkpoint_hashes[model_name] = file_sha256(checkpoint_path)
        save_confusion_matrix(
            matrix,
            ACTIVITY_LABELS,
            results_root / f"{model_name}_confusion_matrix.png",
            f"{model_name.upper()} activity confusion matrix",
        )

    write_metrics_json(
        results_root / "metrics.json",
        {
            "dataset": "HMDB51",
            "fold": metadata.get("fold"),
            "split": "official test",
            "labels": list(ACTIVITY_LABELS),
            "label_to_index": dict(LABEL_TO_INDEX),
            "averaging": "macro",
            "models": results,
        },
    )
    write_comparison_csv(results_root / "model_comparison.csv", results)
    write_predictions_csv(results_root / "predictions.csv", prediction_rows)
    write_metrics_json(
        results_root / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "HMDB51",
            "fold": metadata.get("fold"),
            "labels": list(ACTIVITY_LABELS),
            "label_to_index": dict(LABEL_TO_INDEX),
            "seed": seed,
            "frames_per_sample": metadata.get("frames_per_sample"),
            "sampling_interval_seconds": metadata.get("sampling_interval_seconds"),
            "split_counts": dict(Counter(sample.split for sample in samples)),
            "manifest_sha256": file_sha256(manifest_path),
            "checkpoint_sha256": checkpoint_hashes,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": _torchvision_version(),
            "device": str(device),
            "pretrained_backbones": {
                "cnn": "Torchvision MobileNetV2 ImageNet-1K V2",
                "advanced": "Torchvision S3D Kinetics-400 V1",
            },
            "preprocessing": {
                "sampled_frames": (
                    "fixed source-time grid; hold last observed frame; repeat final frame for short clips"
                    if metadata.get("sampling_interval_seconds") is not None
                    else "uniform source-order sampling (legacy cache)"
                ),
                "person_crop": "largest YOLO Pose detection with 8% padding",
                "mlp": f"{metadata['frames_per_sample']}x17 bounding-box-relative x/y/visibility",
                "cnn": f"mean of {metadata['frames_per_sample']} frozen MobileNetV2 frame features",
                "advanced": f"one {metadata['frames_per_sample']}-frame frozen S3D clip feature",
            },
            "latency": {
                "unit": "milliseconds per video sample",
                "comparable_end_to_end": False,
                "common_exclusions": [
                    "video decoding",
                    "cache file loading",
                ],
                "boundaries": {
                    "mlp": (
                        "classifier inference from cached YOLO pose features; "
                        "YOLO pose estimation is excluded"
                    ),
                    "cnn": (
                        "cached-crop tensor preprocessing, frozen MobileNetV2 "
                        "inference, and classifier inference"
                    ),
                    "advanced": (
                        "cached-clip tensor preprocessing, frozen S3D inference, "
                        "and classifier inference"
                    ),
                },
                "samples_per_model": min(latency_samples, len(test_samples)),
            },
        },
    )
    return results


def validate_checkpoint_labels(checkpoint: dict, checkpoint_path: Path) -> None:
    labels = tuple(checkpoint.get("labels", ()))
    label_to_index = checkpoint.get("label_to_index")
    if labels != ACTIVITY_LABELS or label_to_index != LABEL_TO_INDEX:
        raise ValueError(
            f"Checkpoint label metadata does not match canonical labels: {checkpoint_path}"
        )


def measure_inference_latency(
    model_name: str,
    classifier: torch.nn.Module,
    checkpoint: dict[str, Any],
    samples: Sequence[ActivitySample],
    device: torch.device,
) -> float:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported activity model: {model_name}")
    if not samples:
        raise ValueError("At least one sample is required for latency measurement")
    extractor = None
    if model_name == "cnn":
        extractor = build_mobilenet_extractor().to(device)
    elif model_name == "advanced":
        extractor = build_s3d_extractor().to(device)
    durations: list[float] = []
    with torch.inference_mode():
        for sample in samples:
            with np.load(Path(sample.cache_path), allow_pickle=False) as payload:
                pose = np.asarray(payload["pose"])
                crops = np.asarray(payload["crops"])
            _synchronize(device)
            started = perf_counter()
            if model_name == "mlp":
                raw_features = torch.from_numpy(pose.reshape(1, -1)).float().to(device)
            elif model_name == "cnn":
                assert extractor is not None
                raw_features = extractor(image_tensor(crops).to(device)).mean(
                    dim=0,
                    keepdim=True,
                )
            else:
                assert extractor is not None
                raw_features = extractor(video_tensor(crops).to(device))
            mean = checkpoint["feature_mean"].to(device)
            standard_deviation = checkpoint["feature_std"].to(device)
            classifier((raw_features - mean) / standard_deviation)
            _synchronize(device)
            durations.append((perf_counter() - started) * 1000.0)
    return float(np.mean(durations))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardize(features: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    mean = checkpoint["feature_mean"].cpu().numpy()
    standard_deviation = checkpoint["feature_std"].cpu().numpy()
    return ((features - mean) / standard_deviation).astype(np.float32)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _torchvision_version() -> str:
    import torchvision

    return torchvision.__version__
