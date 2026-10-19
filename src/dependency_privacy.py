"""Dependency opt-outs; these are not an operating-system network sandbox."""

import os

from config import DEFAULT_CONFIG
from data_policy import ensure_private_directory
from pathlib import Path


def disable_ultralytics_telemetry() -> None:
    # Set before import: suppress its connectivity check and Sentry startup.
    # Application-managed, hash-verified downloads remain available.
    os.environ["YOLO_OFFLINE"] = "true"
    cache = DEFAULT_CONFIG.project_root / ".tmp" / "ultralytics"
    cache = Path(os.environ.get("YOLO_CONFIG_DIR", str(cache)))
    ensure_private_directory(cache)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
    from ultralytics import settings
    from ultralytics.utils.events import events

    settings.update({"sync": False})
    # The collector may already exist if a library user imported YOLO first.
    events.enabled = False


def disable_onnx_telemetry() -> None:
    import onnxruntime

    onnxruntime.disable_telemetry_events()
