"""Feature caching and fast classifier-head training."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from data_policy import ensure_private_directory, atomic_output
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activity_recognition.dataset import (
    ActivitySample,
    load_manifest,
    samples_for_split,
)
from activity_recognition.labels import ACTIVITY_LABELS, LABEL_TO_INDEX
from activity_recognition.metrics import (
    classification_metrics,
    confusion_matrix,
    save_training_curves,
)
from activity_recognition.models import (
    MODEL_NAMES,
    build_activity_classifier,
    build_mobilenet_extractor,
    build_s3d_extractor,
)
from activity_recognition.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    KEYPOINT_COUNT,
    S3D_MEAN,
    S3D_STD,
    file_fingerprint,
    image_tensor,
    load_cached_arrays,
    validate_cached_samples,
    video_tensor,
)


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    checkpoint_path: Path
    training_time_seconds: float
    best_validation_macro_f1: float
    epochs_completed: int
    history: dict[str, list[float]]


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def cache_backbone_features(
    samples: Iterable[ActivitySample],
    feature_root: Path,
    model_name: str,
    device: torch.device,
    *,
    overwrite: bool = False,
) -> None:
    """Cache frozen MobileNetV2 or S3D vectors once per source video."""
    if model_name not in {"cnn", "advanced"}:
        raise ValueError("Backbone features are only used by cnn and advanced")
    model_feature_root = feature_root / model_name
    ensure_private_directory(model_feature_root)
    sample_list = list(samples)
    provenance = {sample.key: _feature_provenance(sample, model_name) for sample in sample_list}
    pending_samples = [
        sample
        for sample in sample_list
        if overwrite
        or not _valid_feature_cache(
            model_feature_root / f"{sample.key}.npy", provenance[sample.key]
        )
    ]
    if not pending_samples:
        return

    extractor = (build_mobilenet_extractor() if model_name == "cnn" else build_s3d_extractor()).to(
        device
    )
    with torch.inference_mode():
        for position, sample in enumerate(pending_samples, start=1):
            output_path = model_feature_root / f"{sample.key}.npy"
            with np.load(Path(sample.cache_path), allow_pickle=False) as payload:
                crops = np.asarray(payload["crops"])
            if model_name == "cnn":
                inputs = image_tensor(crops).to(device)
                vector = extractor(inputs).mean(dim=0)
            else:
                inputs = video_tensor(crops).to(device)
                vector = extractor(inputs).squeeze(0)
            with atomic_output(output_path) as temporary_path, temporary_path.open("wb") as stream:
                np.save(
                    stream,
                    vector.detach().cpu().numpy().astype(np.float32),
                    allow_pickle=False,
                )
            sidecar_path = output_path.with_suffix(".json")
            metadata = dict(provenance[sample.key])
            metadata["vector_sha256"] = file_fingerprint(output_path)
            metadata["vector_shape"] = list(vector.shape)
            with atomic_output(sidecar_path) as temporary_sidecar:
                temporary_sidecar.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            print(f"[{position}/{len(pending_samples)}] cached {model_name} features")


def _feature_provenance(sample: ActivitySample, model_name: str) -> dict[str, Any]:
    return {
        "version": 1,
        "source_sha256": file_fingerprint(Path(sample.cache_path)),
        "extractor": "mobilenet_v2.IMAGENET1K_V2" if model_name == "cnn" else "s3d.KINETICS400_V1",
        "torchvision_version": version("torchvision"),
        "torch_version": str(torch.__version__),
        "preprocessing": "rgb-uint8-normalize-v1",
        "mean": list(IMAGENET_MEAN if model_name == "cnn" else S3D_MEAN),
        "std": list(IMAGENET_STD if model_name == "cnn" else S3D_STD),
        "aggregation": "frame-mean" if model_name == "cnn" else "clip",
    }


def _valid_feature_cache(path: Path, expected: dict[str, Any]) -> bool:
    try:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return False
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("vector_sha256") != file_fingerprint(path):
            return False
        vector = np.load(path, allow_pickle=False)
        return (
            vector.ndim == 1
            and vector.size > 0
            and vector.dtype == np.float32
            and list(vector.shape) == metadata.get("vector_shape")
            and bool(np.isfinite(vector).all())
        )
    except (OSError, ValueError, TypeError, EOFError):
        return False


def load_model_features(
    samples: list[ActivitySample],
    model_name: str,
    feature_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported activity model: {model_name}")
    if model_name == "mlp":
        poses, labels = load_cached_arrays(samples, "pose")
        return poses.reshape(len(poses), -1).astype(np.float32), labels
    feature_paths = [feature_root / model_name / f"{sample.key}.npy" for sample in samples]
    missing_paths = [path for path in feature_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Activity {model_name} feature missing: {missing_paths[0]}. "
            "Run train_activity_models.py to build the frozen feature cache."
        )
    vectors = [np.load(path, allow_pickle=False) for path in feature_paths]
    labels = np.asarray([sample.label_index for sample in samples], dtype=np.int64)
    return np.stack(vectors).astype(np.float32), labels


def train_all_models(
    manifest_path: Path,
    feature_root: Path,
    model_root: Path,
    results_root: Path,
    *,
    epochs: int = 20,
    patience: int = 4,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 2026,
    device_name: str = "auto",
    overwrite_features: bool = False,
) -> dict[str, TrainingResult]:
    metadata, samples = load_manifest(manifest_path)
    train_samples = samples_for_split(samples, "train")
    validation_samples = samples_for_split(samples, "validation")
    if not train_samples or not validation_samples:
        raise ValueError("Manifest must contain train and validation samples")
    development_samples = [*train_samples, *validation_samples]
    validate_cached_samples(development_samples, metadata)
    device = resolve_device(device_name)
    set_reproducible_seed(seed)
    ensure_private_directory(model_root)
    ensure_private_directory(feature_root)

    for model_name in ("cnn", "advanced"):
        cache_backbone_features(
            development_samples,
            feature_root,
            model_name,
            device,
            overwrite=overwrite_features,
        )

    results: dict[str, TrainingResult] = {}
    histories: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        training_features, training_labels = load_model_features(
            train_samples,
            model_name,
            feature_root,
        )
        validation_features, validation_labels = load_model_features(
            validation_samples,
            model_name,
            feature_root,
        )
        result = train_classifier(
            model_name,
            training_features,
            training_labels,
            validation_features,
            validation_labels,
            model_root / f"{model_name}.pt",
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            manifest_metadata=metadata,
        )
        results[model_name] = result
        histories[model_name] = result.history
    save_training_curves(histories, results_root / "training_curves.png")
    return results


def train_classifier(
    model_name: str,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    checkpoint_path: Path,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    manifest_metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if training_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("Training and validation features must be 2D arrays")
    if training_features.shape[1] != validation_features.shape[1]:
        raise ValueError("Training and validation feature widths must match")
    frame_count = (manifest_metadata or {}).get("frames_per_sample", 16)
    if model_name == "mlp" and training_features.shape[1] != frame_count * KEYPOINT_COUNT * 3:
        raise ValueError("MLP feature width does not match manifest frame count")
    if len(training_features) != len(training_labels) or len(validation_features) != len(
        validation_labels
    ):
        raise ValueError("Feature and label counts must match")
    if not len(training_features) or not len(validation_features):
        raise ValueError("Training and validation arrays must not be empty")
    if not np.isfinite(training_features).all() or not np.isfinite(validation_features).all():
        raise ValueError("Training and validation features must be finite")
    class_count = len(ACTIVITY_LABELS)
    if (
        np.any(training_labels < 0)
        or np.any(training_labels >= class_count)
        or np.any(validation_labels < 0)
        or np.any(validation_labels >= class_count)
    ):
        raise ValueError("Training and validation labels are out of range")

    mean = training_features.mean(axis=0, keepdims=True)
    standard_deviation = training_features.std(axis=0, keepdims=True)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    training_features = (training_features - mean) / standard_deviation
    validation_features = (validation_features - mean) / standard_deviation

    model = build_activity_classifier(model_name, training_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training_features).float(),
            torch.from_numpy(training_labels).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(validation_features).float().to(device)
    validation_y = torch.from_numpy(validation_labels).long().to(device)
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_macro_f1": [],
    }
    best_score = -1.0
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    started = perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, labels in training_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            sample_count += len(labels)

        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x)
            validation_loss = float(loss_function(validation_logits, validation_y).item())
            validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
        matrix = confusion_matrix(validation_labels, validation_predictions)
        macro_f1 = classification_metrics(matrix)["macro_f1"]
        history["train_loss"].append(running_loss / max(1, sample_count))
        history["validation_loss"].append(validation_loss)
        history["validation_macro_f1"].append(macro_f1)
        print(f"{model_name} epoch {epoch + 1}: loss={validation_loss:.4f} macro_f1={macro_f1:.4f}")

        improved = macro_f1 > best_score or (
            np.isclose(macro_f1, best_score) and validation_loss < best_loss
        )
        if improved:
            best_score = macro_f1
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    training_time = perf_counter() - started
    if best_state is None:
        raise RuntimeError(f"{model_name} training did not produce a checkpoint")
    ensure_private_directory(checkpoint_path.parent)
    with atomic_output(checkpoint_path) as temporary_checkpoint:
        torch.save(
            {
                "model_name": model_name,
                "input_dim": int(training_features.shape[1]),
                "state_dict": best_state,
                "feature_mean": torch.from_numpy(mean.squeeze(0)).float(),
                "feature_std": torch.from_numpy(standard_deviation.squeeze(0)).float(),
                "training_time_seconds": training_time,
                "best_validation_macro_f1": best_score,
                "epochs_completed": len(history["train_loss"]),
                "seed": seed,
                "labels": list(ACTIVITY_LABELS),
                "label_to_index": dict(LABEL_TO_INDEX),
                "frames_per_sample": int((manifest_metadata or {}).get("frames_per_sample", 16)),
                **(
                    {"sampling_interval_seconds": manifest_metadata["sampling_interval_seconds"]}
                    if manifest_metadata and "sampling_interval_seconds" in manifest_metadata
                    else {}
                ),
            },
            temporary_checkpoint,
        )
    return TrainingResult(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        training_time_seconds=training_time,
        best_validation_macro_f1=best_score,
        epochs_completed=len(history["train_loss"]),
        history=history,
    )
