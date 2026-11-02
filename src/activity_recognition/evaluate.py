"""Evaluate trained activity heads on the untouched HMDB51 test split."""

from __future__ import annotations

import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from data_policy import ensure_private_directory, safe_file_path
from time import perf_counter
from typing import Any, Sequence, Callable

import numpy as np
import torch

from activity_recognition.dataset import (
    ActivitySample,
    load_manifest,
    samples_for_split,
)
from activity_recognition.labels import ACTIVITY_LABELS, LABEL_TO_INDEX
from activity_recognition.provenance import (
    require_fresh_outputs,
    environment,
    file_hash,
    cache_quality_report,
)
from activity_recognition.evidence import reconstruct_predictions
from activity_recognition.metrics import (
    classification_metrics,
    confusion_matrix,
    save_confusion_matrix,
    write_comparison_csv,
    write_metrics_json,
    write_predictions_csv,
    write_latency_csv,
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
    feature_contract,
    load_model_features,
    resolve_device,
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
    latency_warmup: int = 3,
    latency_repeats: int = 10,
    include_source_latency: bool = False,
) -> dict[str, dict[str, Any]]:
    if latency_samples < 1:
        raise ValueError("latency_samples must be positive")
    if latency_warmup < 1 or latency_repeats < 2:
        raise ValueError("Latency requires at least one warmup and two repeats")
    metadata, samples = load_manifest(manifest_path)
    test_samples = samples_for_split(samples, "test")
    if not test_samples:
        raise ValueError("Manifest contains no official test samples")
    validate_cached_samples(samples, metadata)
    device = resolve_device(device_name)
    seed = int(metadata.get("seed", 2026))
    require_fresh_outputs(
        results_root,
        [
            "metrics.json",
            "model_comparison.csv",
            "predictions.csv",
            "run_manifest.json",
            "reconstructed_metrics.json",
            "latency.csv",
            *[f"{name}_confusion_matrix.png" for name in MODEL_NAMES],
        ],
    )
    ensure_private_directory(results_root)
    results: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    training_configs: dict[str, Any] = {}
    prediction_rows: list[dict] = []

    for model_name in MODEL_NAMES:
        checkpoint_path = model_root / f"{model_name}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Activity checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            safe_file_path(checkpoint_path),
            map_location=device,
            weights_only=True,
        )
        validate_checkpoint_labels(checkpoint, checkpoint_path)
        if checkpoint.get("dataset_provenance", {}).get("manifest_sha256") != file_hash(
            manifest_path
        ):
            raise ValueError(
                "Checkpoint training manifest differs; use the exact training manifest"
            )
        if checkpoint.get("dataset_provenance", {}).get("cache_provenance") != metadata.get(
            "cache_provenance"
        ) or checkpoint.get("dataset_provenance", {}).get("extraction") != metadata.get(
            "extraction"
        ):
            raise ValueError(
                "Checkpoint dataset/preprocessing provenance differs; use the training manifest"
            )
        if checkpoint.get("model_name") != model_name:
            raise ValueError(f"Checkpoint model name does not match filename: {checkpoint_path}")
        if checkpoint.get("frames_per_sample") != metadata.get("frames_per_sample"):
            raise ValueError(f"Checkpoint frame count does not match manifest: {checkpoint_path}")
        if checkpoint.get("sampling_interval_seconds") != metadata.get("sampling_interval_seconds"):
            raise ValueError(
                f"Checkpoint sampling interval does not match manifest: {checkpoint_path}"
            )
        if model_name in {"cnn", "advanced"}:
            if checkpoint["dataset_provenance"].get("backbone_contracts", {}).get(
                model_name
            ) != feature_contract(model_name):
                raise ValueError(
                    "Checkpoint backbone weights/preprocessing differ from current extraction"
                )
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
        metrics["latency_breakdown"] = measure_latency_breakdown(
            model_name,
            model,
            checkpoint,
            test_samples[:latency_samples],
            device,
            warmup=latency_warmup,
            repeats=latency_repeats,
            source_metadata=metadata if include_source_latency else None,
        )
        metrics["inference_latency_ms"] = metrics["latency_breakdown"]["cached_input_pipeline"][
            "mean_ms"
        ]
        metrics["latency_scope"] = (
            "tensor conversion, feature standardization and classifier from cached pose; excludes YOLO"
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
        training_configs[model_name] = {
            key: checkpoint.get(key)
            for key in ("training_config", "seed", "model_seed", "environment")
        }
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
    write_latency_csv(results_root / "latency.csv", results)
    write_predictions_csv(results_root / "predictions.csv", prediction_rows)
    write_metrics_json(
        results_root / "reconstructed_metrics.json",
        reconstruct_predictions(results_root / "predictions.csv", samples),
    )
    write_metrics_json(
        results_root / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_environment": environment(device),
            "dataset_manifest": metadata,
            "dataset_quality": cache_quality_report(samples),
            "prediction_sha256": file_sha256(results_root / "predictions.csv"),
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
            "training_runs": training_configs,
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
                "cached_input_comparable_end_to_end": False,
                "source_pipeline_measured": include_source_latency,
                "cached_input_exclusions": [
                    "video decoding",
                    "cache file loading",
                    "pose extraction and crop creation",
                ],
                "boundaries": {
                    "mlp": (
                        "tensor conversion, feature standardization and classifier from cached pose; "
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
                "sample_keys": [sample.key for sample in test_samples[:latency_samples]],
                "selection": "first test samples in manifest order; not randomized",
                "warmup_per_sample": latency_warmup,
                "repeats_per_sample": latency_repeats,
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
    """Compatibility scalar; prefer the explicitly scoped breakdown report."""
    return measure_latency_breakdown(model_name, classifier, checkpoint, samples, device)[
        "cached_input_pipeline"
    ]["mean_ms"]


def measure_latency_breakdown(
    model_name: str,
    classifier: torch.nn.Module,
    checkpoint: dict[str, Any],
    samples: Sequence[ActivitySample],
    device: torch.device,
    *,
    warmup: int = 3,
    repeats: int = 10,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES or not samples:
        raise ValueError("Valid model and at least one sample are required")
    if warmup < 1 or repeats < 2:
        raise ValueError("Latency requires at least one warmup and two repeats")
    extractor = None
    if model_name == "cnn":
        extractor = build_mobilenet_extractor().to(device)
    elif model_name == "advanced":
        extractor = build_s3d_extractor().to(device)
    durations: dict[str, list[float]] = {
        key: []
        for key in ("preprocessing", "backbone_only", "classifier_only", "cached_input_pipeline")
    }
    mean, std = checkpoint["feature_mean"].to(device), checkpoint["feature_std"].to(device)
    classifier.eval()
    frame_batch_sizes: set[int] = set()
    with torch.inference_mode():
        for sample in samples:
            with np.load(safe_file_path(Path(sample.cache_path)), allow_pickle=False) as payload:
                pose, crops = np.asarray(payload["pose"]), np.asarray(payload["crops"])
            frame_batch_sizes.add(int(pose.shape[0]) if model_name == "cnn" else 1)
            for iteration in range(warmup + repeats):
                _synchronize(device)
                start = perf_counter()
                if model_name == "mlp":
                    inputs = torch.from_numpy(pose.reshape(1, -1)).float().to(device)
                elif model_name == "cnn":
                    inputs = image_tensor(crops).to(device)
                else:
                    inputs = video_tensor(crops).to(device)
                _synchronize(device)
                prepared = perf_counter()
                if extractor is None:
                    raw = inputs
                elif model_name == "cnn":
                    raw = extractor(inputs).mean(dim=0, keepdim=True)
                else:
                    raw = extractor(inputs)
                _synchronize(device)
                extracted = perf_counter()
                normalized = (raw - mean) / std
                _synchronize(device)
                standardized = perf_counter()
                classifier(normalized)
                _synchronize(device)
                finished = perf_counter()
                if iteration >= warmup:
                    durations["preprocessing"].append(
                        ((prepared - start) + (standardized - extracted)) * 1000
                    )
                    durations["backbone_only"].append((extracted - prepared) * 1000)
                    durations["classifier_only"].append((finished - standardized) * 1000)
                    durations["cached_input_pipeline"].append((finished - start) * 1000)
    result: dict[str, Any] = {key: latency_statistics(values) for key, values in durations.items()}
    if model_name == "mlp":
        result["backbone_only"] = None
    result.update(
        {
            "pose_extraction": None,
            "end_to_end_video": None,
            "unmeasured_reason": "Cached-input measurement excludes video decoding, YOLO, person selection and crop creation.",
            "warmup_per_sample": warmup,
            "repeats_per_sample": repeats,
            "video_batch_size": 1,
            "backbone_frame_batch_sizes": sorted(frame_batch_sizes),
            "synchronization": "selected CUDA device or MPS synchronized at stage boundaries; CPU synchronous",
            "preprocessing_scope": "tensor conversion, normalization, device transfer and feature standardization",
            "backbone_scope": "frozen extractor forward plus frame pooling; excludes classifier",
            "classifier_scope": "head forward only; device-resident standardized inputs",
            "cached_input_pipeline_scope": "tensor preprocessing, backbone/pooling, standardization, head; includes synchronization overhead",
        }
    )
    if source_metadata is not None:
        from activity_recognition.provenance import pose_extraction_provenance
        from config import AppConfig
        from pose import PoseAnalyzer

        if source_metadata["extraction"] != pose_extraction_provenance():
            raise ValueError("Current pose extraction differs from the experiment manifest")
        config = AppConfig.from_env()
        analyzer = PoseAnalyzer(config=config, model_path=config.pose_model_path)
        analyzer.model.to(device)

        def predict_clip(pose: np.ndarray, crops: np.ndarray) -> None:
            if model_name == "mlp":
                raw = torch.from_numpy(pose.reshape(1, -1)).float().to(device)
            elif model_name == "cnn":
                assert extractor is not None
                raw = extractor(image_tensor(crops).to(device)).mean(dim=0, keepdim=True)
            else:
                assert extractor is not None
                raw = extractor(video_tensor(crops).to(device))
            classifier((raw - mean) / std)

        result.update(
            measure_source_latency(
                samples,
                analyzer,
                predict_clip,
                device,
                frames_per_sample=source_metadata["frames_per_sample"],
                interval=source_metadata["sampling_interval_seconds"],
                warmup=warmup,
                repeats=repeats,
            )
        )
        result["unmeasured_reason"] = None
    return result


def measure_source_latency(
    samples: Sequence[ActivitySample],
    analyzer: Any,
    predict_clip: Callable[[np.ndarray, np.ndarray], None],
    device: torch.device,
    *,
    frames_per_sample: int,
    interval: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from activity_recognition.preprocessing import sample_video_frames, normalize_pose, crop_person

    if not samples or warmup < 1 or repeats < 2:
        raise ValueError("Source latency requires samples, warmups and repeated measurements")
    durations: dict[str, list[float]] = {
        key: []
        for key in (
            "video_decode_sampling",
            "pose_extraction",
            "person_selection_crop_pose_normalization",
            "end_to_end_video",
        )
    }
    with torch.inference_mode():
        for sample in samples:
            for iteration in range(warmup + repeats):
                analyzer.reset_tracking()
                _synchronize(device)
                start = perf_counter()
                frames = sample_video_frames(
                    Path(sample.video_path), frames_per_sample, sampling_interval_seconds=interval
                )
                decoded = perf_counter()
                pose_ms = 0.0
                prepare_ms = 0.0
                poses, crops = [], []
                for frame in frames:
                    _synchronize(device)
                    before = perf_counter()
                    results = analyzer.analyze(frame)
                    _synchronize(device)
                    after = perf_counter()
                    pose_ms += (after - before) * 1000
                    selected = max(
                        results,
                        key=lambda r: (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]),
                        default=None,
                    )
                    box = selected.box if selected is not None else None
                    poses.append(
                        normalize_pose(
                            selected.landmarks if selected is not None else {}, box, frame.shape
                        )
                    )
                    crops.append(crop_person(frame, box))
                    prepare_ms += (perf_counter() - after) * 1000
                predict_clip(np.stack(poses), np.stack(crops))
                _synchronize(device)
                finish = perf_counter()
                if iteration >= warmup:
                    durations["video_decode_sampling"].append((decoded - start) * 1000)
                    durations["pose_extraction"].append(pose_ms)
                    durations["person_selection_crop_pose_normalization"].append(prepare_ms)
                    durations["end_to_end_video"].append((finish - start) * 1000)
    return {
        **{key: latency_statistics(values) for key, values in durations.items()},
        "source_pipeline_scope": "decode fixed sample window, YOLO predictor initialization and backend warmup on every clip, tracking, largest-person selection, crop/pose normalization, tensor preprocessing, optional backbone, standardization and head; excludes initial weight loading, reset call, rendering and encoding",
        "pose_device_policy": f"pose model explicitly moved to classifier device: {device}",
        "source_video_batch_size": 1,
        "source_warmup_per_sample": warmup,
        "source_repeats_per_sample": repeats,
    }


def latency_statistics(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "measurements_ms": list(values),
    }


def file_sha256(path: Path) -> str:
    return file_hash(path)


def _standardize(features: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    mean = checkpoint["feature_mean"].cpu().numpy()
    standard_deviation = checkpoint["feature_std"].cpu().numpy()
    return ((features - mean) / standard_deviation).astype(np.float32)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _torchvision_version() -> str:
    import torchvision

    return torchvision.__version__
