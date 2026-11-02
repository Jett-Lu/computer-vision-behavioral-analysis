"""HMDB51 fold parsing and cached activity-sample metadata."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from data_policy import ensure_private_directory, atomic_output, _safe_path, safe_file_path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from activity_recognition.labels import (
    ACTIVITY_LABELS,
    HMDB51_TO_ACTIVITY,
    LABEL_TO_INDEX,
    normalize_hmdb51_label,
)
from activity_recognition.sampling import (
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    validate_sampling_interval,
)

MANIFEST_VERSION = 1
DEFAULT_HMDB51_MIRROR = "https://huggingface.co/datasets/Sina272/hmdb51-v2/resolve/main"


@dataclass(frozen=True)
class ActivitySample:
    key: str
    video_path: str
    cache_path: str
    label: str
    label_index: int
    split: str
    hmdb51_class: str


def deterministic_validation_keys(
    video_names: Iterable[str],
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> set[str]:
    """Select a stable validation subset without splitting frames or clips."""
    names = sorted(set(video_names))
    if not names:
        return set()
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_count = min(
        len(names) - 1,
        max(1, round(len(names) * validation_fraction)),
    )
    ranked = sorted(
        names,
        key=lambda name: sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[:validation_count])


def build_hmdb51_fold(
    dataset_root: Path,
    annotations_root: Path,
    cache_root: Path,
    *,
    fold: int = 1,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> list[ActivitySample]:
    """Build train/validation/test records from an official HMDB51 fold."""
    if fold not in {1, 2, 3}:
        raise ValueError("HMDB51 fold must be 1, 2, or 3")
    dataset_root = _safe_path(dataset_root)
    annotations_root = annotations_root.resolve()
    cache_root = _safe_path(cache_root)
    samples: list[ActivitySample] = []

    for hmdb_class, activity_label in HMDB51_TO_ACTIVITY.items():
        split_path = annotations_root / f"{hmdb_class}_test_split{fold}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"HMDB51 split file not found: {split_path}")
        rows = _read_split_file(split_path)
        training_names = [name for name, flag in rows if flag == 1]
        validation_names = deterministic_validation_keys(
            training_names,
            validation_fraction,
            seed,
        )

        for video_name, flag in rows:
            if flag == 0:
                continue
            split = (
                "test" if flag == 2 else "validation" if video_name in validation_names else "train"
            )
            video_path = _find_video(dataset_root, hmdb_class, video_name)
            key = sha256(f"{hmdb_class}/{video_name}".encode("utf-8")).hexdigest()[:16]
            samples.append(
                ActivitySample(
                    key=key,
                    video_path=str(video_path),
                    cache_path=str(cache_root / "samples" / f"{key}.npz"),
                    label=activity_label,
                    label_index=LABEL_TO_INDEX[activity_label],
                    split=split,
                    hmdb51_class=hmdb_class,
                )
            )
    return sorted(samples, key=lambda sample: (sample.split, sample.label, sample.key))


def build_hmdb51_metadata_fold(
    dataset_root: Path,
    training_metadata_path: Path,
    test_metadata_path: Path,
    cache_root: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> list[ActivitySample]:
    """Build fold-1 records from mirror metadata preserving its test split."""
    dataset_root = _safe_path(dataset_root)
    cache_root = _safe_path(cache_root)
    training_rows = _read_metadata_csv(training_metadata_path)
    test_rows = _read_metadata_csv(test_metadata_path)
    samples: list[ActivitySample] = []
    for hmdb_class, activity_label in HMDB51_TO_ACTIVITY.items():
        class_training = [row for row in training_rows if row["label"] == hmdb_class]
        class_test = [row for row in test_rows if row["label"] == hmdb_class]
        training_names = {row["file_name"] for row in class_training}
        test_names = {row["file_name"] for row in class_test}
        overlap = training_names & test_names
        if overlap:
            duplicate = sorted(overlap)[0]
            raise ValueError(f"HMDB51 metadata places a video in train and test: {duplicate}")
        validation_names = deterministic_validation_keys(
            training_names,
            validation_fraction,
            seed,
        )
        split_rows = [(row, False) for row in class_training] + [(row, True) for row in class_test]
        for row, is_test in split_rows:
            relative_path = _relative_video_path(row["file_name"])
            video_path = dataset_root / relative_path
            if not video_path.is_file():
                raise FileNotFoundError(f"HMDB51 video not found: {video_path}")
            split = (
                "test"
                if is_test
                else "validation"
                if row["file_name"] in validation_names
                else "train"
            )
            key = sha256(row["file_name"].encode("utf-8")).hexdigest()[:16]
            samples.append(
                ActivitySample(
                    key=key,
                    video_path=str(video_path.resolve()),
                    cache_path=str(cache_root / "samples" / f"{key}.npz"),
                    label=activity_label,
                    label_index=LABEL_TO_INDEX[activity_label],
                    split=split,
                    hmdb51_class=hmdb_class,
                )
            )
    return sorted(samples, key=lambda sample: (sample.split, sample.label, sample.key))


def download_hmdb51_subset(
    metadata_paths: Iterable[Path],
    dataset_root: Path,
    *,
    base_url: str = DEFAULT_HMDB51_MIRROR,
    overwrite: bool = False,
) -> dict[str, int]:
    """Download only the four target classes listed by fold-1 metadata."""
    dataset_root = _safe_path(dataset_root)
    rows = [
        row
        for metadata_path in metadata_paths
        for row in _read_metadata_csv(metadata_path)
        if row["label"] in HMDB51_TO_ACTIVITY
    ]
    downloaded = skipped = failed = 0
    for position, row in enumerate(rows, start=1):
        relative_path = _relative_video_path(row["file_name"])
        output_path = dataset_root / relative_path
        if output_path.is_file() and not overwrite:
            skipped += 1
            continue
        ensure_private_directory(output_path.parent)
        url = f"{base_url.rstrip('/')}/{quote(row['file_name'], safe='/')}?download=true"
        request = Request(url, headers={"User-Agent": "activity-benchmark/1.0"})
        try:
            with atomic_output(output_path) as temporary_path:
                with urlopen(request, timeout=60) as response:
                    with temporary_path.open("wb") as destination:
                        shutil.copyfileobj(response, destination)
                if temporary_path.stat().st_size == 0:
                    raise RuntimeError("download was empty")
            downloaded += 1
            print(f"[{position}/{len(rows)}] downloaded {relative_path}")
        except (OSError, RuntimeError, URLError) as exc:
            failed += 1
            print(f"[{position}/{len(rows)}] failed {relative_path}: {exc}")
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def write_manifest(
    path: Path,
    samples: list[ActivitySample],
    *,
    dataset_root: Path,
    annotations_root: Path,
    fold: int,
    validation_fraction: float,
    seed: int,
    frames_per_sample: int,
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    extraction_provenance: dict[str, Any] | None = None,
    split_definition: dict[str, Any] | None = None,
) -> None:
    ensure_private_directory(path.parent)
    payload = {
        "version": MANIFEST_VERSION,
        "dataset": "HMDB51",
        "dataset_root": str(dataset_root.resolve()),
        "annotations_root": str(annotations_root.resolve()),
        "fold": fold,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "frames_per_sample": frames_per_sample,
        "sampling_interval_seconds": validate_sampling_interval(sampling_interval_seconds),
        "labels": list(ACTIVITY_LABELS),
        "label_to_index": dict(LABEL_TO_INDEX),
        "samples": [asdict(sample) for sample in samples],
    }
    if extraction_provenance is not None:
        from activity_recognition.provenance import dataset_provenance

        payload["extraction"] = extraction_provenance
        payload["cache_provenance"] = dataset_provenance(samples)
    if split_definition is not None:
        payload["split_definition"] = split_definition
    with atomic_output(path) as temporary_path:
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> tuple[dict[str, Any], list[ActivitySample]]:
    payload = json.loads(safe_file_path(path).read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError("Unsupported activity manifest version")
    labels = payload.get("labels")
    if labels is not None and tuple(labels) != ACTIVITY_LABELS:
        raise ValueError("Activity manifest label order does not match canonical labels")
    label_to_index = payload.get("label_to_index")
    if label_to_index is not None and label_to_index != LABEL_TO_INDEX:
        raise ValueError("Activity manifest label mapping does not match canonical labels")
    frames_per_sample = payload.get("frames_per_sample")
    if not isinstance(frames_per_sample, int) or frames_per_sample < 1:
        raise ValueError("Activity manifest frames_per_sample must be a positive integer")
    if "sampling_interval_seconds" in payload:
        validate_sampling_interval(payload["sampling_interval_seconds"])
    samples = [ActivitySample(**sample) for sample in payload.get("samples", [])]
    if not samples:
        raise ValueError("Activity manifest contains no samples")
    _validate_manifest_samples(samples)
    return payload, samples


def samples_for_split(
    samples: Iterable[ActivitySample],
    split: str,
) -> list[ActivitySample]:
    return [sample for sample in samples if sample.split == split]


def _read_split_file(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    video_names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid HMDB51 split row in {path}: {line}")
        flag = int(parts[1])
        if flag not in {0, 1, 2}:
            raise ValueError(f"Invalid HMDB51 split flag in {path}: {flag}")
        if parts[0] in video_names:
            raise ValueError(f"Duplicate HMDB51 split video in {path}: {parts[0]}")
        video_names.add(parts[0])
        rows.append((parts[0], flag))
    return rows


def _read_metadata_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"file_name", "label"}:
        raise ValueError(f"Invalid HMDB51 metadata CSV: {path}")
    file_names = [row["file_name"] for row in rows]
    if len(file_names) != len(set(file_names)):
        raise ValueError(f"Duplicate HMDB51 video in metadata CSV: {path}")
    return rows


def _find_video(dataset_root: Path, hmdb_class: str, video_name: str) -> Path:
    candidates = (
        dataset_root / hmdb_class / video_name,
        dataset_root / "hmdb51_org" / hmdb_class / video_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"HMDB51 video not found for {hmdb_class}/{video_name} under {dataset_root}"
    )


def _relative_video_path(file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"HMDB51 metadata contains an unsafe video path: {file_name}")
    return path


def _validate_manifest_samples(samples: list[ActivitySample]) -> None:
    valid_splits = {"train", "validation", "test"}
    seen_keys: set[str] = set()
    seen_videos: set[str] = set()
    seen_caches: set[str] = set()
    for sample in samples:
        if not all(
            isinstance(value, str)
            for value in (
                sample.key,
                sample.video_path,
                sample.cache_path,
                sample.label,
                sample.split,
                sample.hmdb51_class,
            )
        ):
            raise ValueError("Activity manifest sample text fields must be strings")
        if sample.split not in valid_splits:
            raise ValueError(f"Activity sample {sample.key} has invalid split: {sample.split}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample.key):
            raise ValueError("Activity sample key must be a safe filename identifier")
        expected_label = normalize_hmdb51_label(sample.hmdb51_class)
        if sample.label != expected_label:
            raise ValueError(
                f"Activity sample {sample.key} label does not match "
                f"HMDB51 class {sample.hmdb51_class}"
            )
        if LABEL_TO_INDEX[sample.label] != sample.label_index:
            raise ValueError(
                f"Activity sample {sample.key} has invalid label index: {sample.label_index}"
            )
        video_path = str(Path(sample.video_path).resolve()).casefold()
        cache_path = str(Path(sample.cache_path).resolve()).casefold()
        if sample.key in seen_keys:
            raise ValueError(f"Duplicate activity sample key: {sample.key}")
        if video_path in seen_videos:
            raise ValueError(f"Duplicate activity source video: {sample.video_path}")
        if cache_path in seen_caches:
            raise ValueError(f"Duplicate activity cache path: {sample.cache_path}")
        seen_keys.add(sample.key)
        seen_videos.add(video_path)
        seen_caches.add(cache_path)
