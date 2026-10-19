"""Operator-facing environment and model readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

from config import AppConfig
from data_policy import private_temporary_file
from model_manager import (
    ModelUnavailableError,
    emotion_model_spec,
    ensure_model,
    file_sha256,
    identity_model_specs,
    pose_model_spec,
)


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    passed: bool
    detail: str


def run_diagnostics(
    config: AppConfig,
    prepare_models: bool = False,
) -> list[DiagnosticCheck]:
    checks = [
        DiagnosticCheck(
            "Platform",
            True,
            f"{platform.system()} {platform.release()} | Python {sys.version.split()[0]}",
        ),
        _check_packages(config),
        _check_writable_directory(config.output_dir, "Output directory"),
        _check_writable_directory(config.data_dir, "Model directory"),
    ]

    specs = [pose_model_spec(config)]
    if config.identity_enabled:
        checks.append(_check_opencv_face_apis())
        specs.extend(identity_model_specs(config))
    if config.expression_enabled:
        checks.append(
            _check_readable_file(config.emotion_face_model_path, "MediaPipe face detector")
        )
        specs.append(emotion_model_spec(config))
    for spec in specs:
        if prepare_models:
            try:
                ensure_model(spec, config)
            except (ModelUnavailableError, OSError, ValueError) as exc:
                checks.append(DiagnosticCheck(spec.name, False, str(exc)))
                continue
        if not spec.path.exists():
            checks.append(DiagnosticCheck(spec.name, False, f"missing: {spec.path}"))
            continue
        try:
            actual_hash = file_sha256(spec.path)
        except OSError as exc:
            checks.append(DiagnosticCheck(spec.name, False, str(exc)))
            continue
        checks.append(
            DiagnosticCheck(
                spec.name,
                actual_hash == spec.sha256,
                "ready" if actual_hash == spec.sha256 else "checksum mismatch",
            )
        )
    return checks


def print_diagnostics(checks: list[DiagnosticCheck]) -> None:
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(f"\nOverall: {'READY' if all(check.passed for check in checks) else 'ACTION REQUIRED'}")


def _check_packages(config: AppConfig | None = None) -> DiagnosticCheck:
    package_names = (
        "opencv-contrib-python",
        "numpy",
        "mediapipe",
        "emotiefflib",
        "ultralytics",
        "lap",
    )
    installed: list[str] = []
    missing: list[str] = []
    for package_name in package_names:
        if (
            config is not None
            and not config.expression_enabled
            and package_name in {"mediapipe", "emotiefflib"}
        ):
            continue
        try:
            installed.append(f"{package_name}={version(package_name)}")
        except PackageNotFoundError:
            missing.append(package_name)
    return DiagnosticCheck(
        "Python packages",
        not missing,
        ", ".join(installed) if not missing else f"missing: {', '.join(missing)}",
    )


def _check_opencv_face_apis() -> DiagnosticCheck:
    try:
        import cv2
    except ImportError as exc:
        return DiagnosticCheck("OpenCV face APIs", False, str(exc))

    available = hasattr(cv2, "FaceDetectorYN_create") and hasattr(
        cv2,
        "FaceRecognizerSF_create",
    )
    return DiagnosticCheck(
        "OpenCV face APIs",
        available,
        "YuNet and SFace APIs available" if available else "Install opencv-contrib-python",
    )


def _check_writable_directory(path: Path, name: str) -> DiagnosticCheck:
    try:
        with private_temporary_file(path) as probe:
            probe.write_text("ok", encoding="utf-8")
        return DiagnosticCheck(name, True, str(path))
    except (OSError, ValueError) as exc:
        return DiagnosticCheck(name, False, str(exc))


def _check_readable_file(path: Path, name: str) -> DiagnosticCheck:
    try:
        with path.open("rb") as source:
            source.read(1)
    except OSError as exc:
        return DiagnosticCheck(name, False, str(exc))
    return DiagnosticCheck(name, True, str(path))
