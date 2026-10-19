"""Manifest-driven end-to-end validation for recorded POC scenarios."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from camera import CaptureClock
from data_policy import atomic_output
from config import AppConfig
from detection import MonitoringProcessor


@dataclass(frozen=True)
class ValidationMetrics:
    frames_processed: int
    max_people: int
    average_people: float
    processing_fps: float
    observed_identities: tuple[str, ...]
    tier_counts: dict[str, int]


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    failures: tuple[str, ...]
    metrics: ValidationMetrics | None


def run_validation(
    manifest_path: Path,
    report_path: Path | None = None,
    config: AppConfig | None = None,
) -> list[ValidationResult]:
    """Process every manifest case and optionally write a machine-readable report."""
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Validation manifest must be a JSON object.")
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Validation manifest must contain a non-empty 'cases' list.")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("Every validation case must be a JSON object.")

    runtime_config = config or AppConfig.from_env()
    processor = MonitoringProcessor(runtime_config)
    results: list[ValidationResult] = []
    try:
        for case in cases:
            try:
                results.append(
                    _run_case(
                        case,
                        manifest_path.parent,
                        processor,
                    )
                )
            except (TypeError, ValueError) as exc:
                results.append(
                    ValidationResult(
                        str(case.get("name") or "unnamed case"),
                        False,
                        (f"invalid case: {exc}",),
                        None,
                    )
                )
    finally:
        processor.close()

    if report_path is not None:
        with atomic_output(report_path) as temporary_path:
            temporary_path.write_text(
                json.dumps(
                    {
                        "passed": all(result.passed for result in results),
                        "results": [asdict(result) for result in results],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    return results


def evaluate_metrics(
    case: dict[str, Any],
    metrics: ValidationMetrics,
) -> tuple[str, ...]:
    """Return human-readable acceptance failures for one measured scenario."""
    failures: list[str] = []
    if metrics.frames_processed == 0:
        failures.append("input contained no readable frames")

    minimum_people = int(case.get("min_people", 0))
    if metrics.max_people < minimum_people:
        failures.append(f"expected at least {minimum_people} people, observed {metrics.max_people}")

    minimum_fps = float(case.get("min_processing_fps", 0.0))
    if metrics.processing_fps < minimum_fps:
        failures.append(
            f"expected at least {minimum_fps:.1f} processing FPS, "
            f"observed {metrics.processing_fps:.1f}"
        )

    observed = set(metrics.observed_identities)
    expected_identities = _identity_set(case, "expected_identities")
    missing = sorted(expected_identities - observed)
    if missing:
        failures.append(f"expected identities not observed: {', '.join(missing)}")

    forbidden_identities = _identity_set(case, "forbidden_identities")
    unexpected = sorted(forbidden_identities & observed)
    if unexpected:
        failures.append(f"forbidden identities observed: {', '.join(unexpected)}")

    maximum_high_ratio = float(case.get("max_high_frame_ratio", 1.0))
    if not 0.0 <= maximum_high_ratio <= 1.0:
        raise ValueError("max_high_frame_ratio must be between 0 and 1")
    high_frames = metrics.tier_counts.get("HIGH", 0)
    high_ratio = high_frames / max(1, metrics.frames_processed)
    if high_ratio > maximum_high_ratio:
        failures.append(
            f"HIGH tier ratio {high_ratio:.2%} exceeds allowed {maximum_high_ratio:.2%}"
        )
    return tuple(failures)


def print_validation_summary(results: list[ValidationResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        if result.metrics is not None:
            print(
                f"  frames={result.metrics.frames_processed} "
                f"people={result.metrics.max_people} "
                f"fps={result.metrics.processing_fps:.1f} "
                f"identities={list(result.metrics.observed_identities)}"
            )
        for failure in result.failures:
            print(f"  - {failure}")


def _run_case(
    case: dict[str, Any],
    manifest_dir: Path,
    processor: MonitoringProcessor,
) -> ValidationResult:
    name = str(case.get("name") or "unnamed case")
    raw_input = case.get("input")
    if not raw_input:
        return ValidationResult(name, False, ("case has no input path",), None)

    input_path = Path(raw_input)
    if not input_path.is_absolute():
        input_path = (manifest_dir / input_path).resolve()
    if not input_path.exists():
        return ValidationResult(name, False, (f"input not found: {input_path}",), None)

    processor.reset_tracking(validation_case=name, source=str(input_path))
    max_frames = max(1, int(case.get("max_frames", 300)))
    frames_processed = 0
    people_total = 0
    max_people = 0
    identities: set[str] = set()
    tiers: Counter[str] = Counter()
    started = perf_counter()

    image = cv2.imread(str(input_path))
    if image is not None:
        warmup_frames = max(1, processor.config.identity_required_matches)
        for frame_number in range(warmup_frames):
            processor.process_frame(
                image,
                timestamp=float(frame_number) / 30.0,
                force_identity=True,
            )
            people_total, max_people = _collect_snapshots(
                processor.last_snapshots,
                identities,
                tiers,
                people_total,
                max_people,
            )
        frames_processed = warmup_frames
    else:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            return ValidationResult(name, False, ("OpenCV could not open input",), None)
        try:
            clock = CaptureClock(capture, recorded=True)
            while frames_processed < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                processor.process_frame(frame, timestamp=clock.timestamp())
                frames_processed += 1
                people_total, max_people = _collect_snapshots(
                    processor.last_snapshots,
                    identities,
                    tiers,
                    people_total,
                    max_people,
                )
        finally:
            capture.release()

    elapsed = max(perf_counter() - started, 1e-6)
    metrics = ValidationMetrics(
        frames_processed=frames_processed,
        max_people=max_people,
        average_people=people_total / max(1, frames_processed),
        processing_fps=frames_processed / elapsed,
        observed_identities=tuple(sorted(identities)),
        tier_counts=dict(tiers),
    )
    failures = evaluate_metrics(case, metrics)
    return ValidationResult(name, not failures, failures, metrics)


def _collect_snapshots(
    snapshots: list[Any],
    identities: set[str],
    tiers: Counter[str],
    people_total: int,
    max_people: int,
) -> tuple[int, int]:
    people_total += len(snapshots)
    max_people = max(max_people, len(snapshots))
    for snapshot in snapshots:
        if snapshot.identity_confirmed:
            identities.add(snapshot.identity_name)
    for tier_label in {snapshot.tier_label for snapshot in snapshots}:
        tiers[tier_label] += 1
    return people_total, max_people


def _identity_set(case: dict[str, Any], field_name: str) -> set[str]:
    values = case.get(field_name, [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} must be a list of names")
    return set(values)
