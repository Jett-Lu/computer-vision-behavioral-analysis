"""Independently reconstruct aggregate metrics from saved per-video predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from activity_recognition.dataset import ActivitySample
from activity_recognition.labels import ACTIVITY_LABELS
from activity_recognition.provenance import file_hash
from data_policy import safe_file_path


def reconstruct_predictions(
    path: Path, samples: Sequence[ActivitySample] | None = None
) -> dict[str, Any]:
    with safe_file_path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Prediction file is empty")
    expected_samples = (
        {sample.key: sample for sample in samples if sample.split == "test"}
        if samples is not None
        else None
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        try:
            model, key = row["model"], row["sample_key"]
            expected, predicted = int(row["expected_index"]), int(row["predicted_index"])
            if (
                not model
                or not key
                or row["split"] != "test"
                or not 0 <= expected < 4
                or not 0 <= predicted < 4
            ):
                raise ValueError("Invalid model/key/split/index")
            if (
                row["expected_label"] != ACTIVITY_LABELS[expected]
                or row["predicted_label"] != ACTIVITY_LABELS[predicted]
            ):
                raise ValueError("Label text does not match index")
            if key in grouped.setdefault(model, {}):
                raise ValueError("Duplicate prediction for model/sample")
            if expected_samples is not None:
                sample = expected_samples.get(key)
                if (
                    sample is None
                    or sample.label_index != expected
                    or Path(sample.video_path).name != row["source_video"]
                ):
                    raise ValueError("Prediction does not match manifest test partition")
            grouped[model][key] = {**row, "expected_index": expected, "predicted_index": predicted}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prediction row: {exc}") from exc
    reference = next(iter(grouped.values()))
    results = {}
    for model, predictions in grouped.items():
        if predictions.keys() != reference.keys() or (
            expected_samples is not None and predictions.keys() != expected_samples.keys()
        ):
            raise ValueError("Missing or inconsistent test predictions")
        if any(
            (row["expected_index"], row["source_video"])
            != (reference[key]["expected_index"], reference[key]["source_video"])
            for key, row in predictions.items()
        ):
            raise ValueError("Models disagree on source/ground truth")
        matrix = [[0] * 4 for _ in range(4)]
        errors = []
        for row in predictions.values():
            matrix[row["expected_index"]][row["predicted_index"]] += 1
            if row["expected_index"] != row["predicted_index"]:
                errors.append(row)
        classes = {}
        for i, label in enumerate(ACTIVITY_LABELS):
            true_positive = matrix[i][i]
            support = sum(matrix[i])
            predicted_count = sum(matrix[j][i] for j in range(4))
            precision = true_positive / predicted_count if predicted_count else 0
            recall = true_positive / support if support else 0
            classes[label] = {
                "support": support,
                "predicted_count": predicted_count,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            }
        results[model] = {
            "sample_count": len(predictions),
            "accuracy": sum(matrix[i][i] for i in range(4)) / len(predictions),
            "macro_precision": sum(c["precision"] for c in classes.values()) / 4,
            "macro_recall": sum(c["recall"] for c in classes.values()) / 4,
            "macro_f1": sum(c["f1"] for c in classes.values()) / 4,
            "confusion_matrix": matrix,
            "per_class": classes,
            "misclassifications": errors,
        }
    return {
        "predictions_sha256": file_hash(path),
        "partition_verified_against_manifest": samples is not None,
        "averaging": "macro, all four canonical classes; zero division = 0",
        "models": results,
    }
