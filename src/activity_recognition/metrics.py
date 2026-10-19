"""Dependency-light classification metrics and report plots."""

from __future__ import annotations

import csv
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
from data_policy import atomic_output


def confusion_matrix(
    expected: Sequence[int] | np.ndarray,
    predicted: Sequence[int] | np.ndarray,
    class_count: int = 4,
) -> np.ndarray:
    if class_count < 1:
        raise ValueError("class_count must be positive")
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted label counts must match")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(expected, predicted):
        if (
            isinstance(target, (bool, np.bool_))
            or not isinstance(target, Integral)
            or isinstance(prediction, (bool, np.bool_))
            or not isinstance(prediction, Integral)
        ):
            raise ValueError("Class labels must be integers")
        if not 0 <= int(target) < class_count:
            raise ValueError(f"Expected class index is out of range: {target}")
        if not 0 <= int(prediction) < class_count:
            raise ValueError(f"Predicted class index is out of range: {prediction}")
        matrix[int(target), int(prediction)] += 1
    return matrix


class ClassificationMetrics(TypedDict):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    sample_count: int


def classification_metrics(matrix: np.ndarray) -> ClassificationMetrics:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("Confusion matrix must be a non-empty square array")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Confusion matrix counts must be integers")
    if np.any(matrix < 0):
        raise ValueError("Confusion matrix counts must not be negative")
    total = int(matrix.sum())
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    expected_count = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        expected_count,
        out=np.zeros_like(true_positive),
        where=expected_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "sample_count": total,
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    with atomic_output(output_path) as temporary:
        figure.savefig(temporary, dpi=160)
    plt.close(figure)


def save_training_curves(histories: dict[str, dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{model_name} train")
        axes[0].plot(
            epochs,
            history["validation_loss"],
            linestyle="--",
            label=f"{model_name} validation",
        )
        axes[1].plot(epochs, history["validation_macro_f1"], label=model_name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(
        title="Validation macro F1",
        xlabel="Epoch",
        ylabel="Macro F1",
        ylim=(0.0, 1.0),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    with atomic_output(output_path) as temporary:
        figure.savefig(temporary, dpi=160)
    plt.close(figure)


def write_metrics_json(path: Path, payload: dict) -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_comparison_csv(path: Path, results: dict[str, dict]) -> None:
    fields = (
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "training_time_seconds",
        "inference_latency_ms",
        "sample_count",
    )
    with (
        atomic_output(path) as temporary,
        temporary.open("w", newline="", encoding="utf-8") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow(
                {"model": model_name, **{field: metrics[field] for field in fields[1:]}}
            )


def write_predictions_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = (
        "model",
        "sample_key",
        "source_video",
        "split",
        "expected_index",
        "expected_label",
        "predicted_index",
        "predicted_label",
    )
    with (
        atomic_output(path) as temporary,
        temporary.open("w", newline="", encoding="utf-8") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
