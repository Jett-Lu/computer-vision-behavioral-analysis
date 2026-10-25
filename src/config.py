"""Central runtime configuration for the monitoring POC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import os

from monitoring_assets import FACE_DETECTOR_PATH


def _runtime_root() -> Path:
    override = os.environ.get("MONITOR_HOME", "").strip()
    if override:
        # Preserve links for the security helpers to reject before resolution.
        return Path(override).expanduser().absolute()
    module_dir = Path(__file__).resolve().parent
    if module_dir.name == "src" and (module_dir.parent / "pyproject.toml").is_file():
        return module_dir.parent
    return Path.home() / ".integrated-monitoring-poc"


PROJECT_ROOT = _runtime_root()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _env_float(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if not math.isfinite(parsed):
        return default
    parsed = max(minimum, parsed)
    return min(parsed, maximum) if maximum is not None else parsed


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().casefold()
    return value if value in choices else default


def _env_path(name: str, default: Path, root: Path = PROJECT_ROOT) -> Path:
    value = os.environ.get(name)
    if value is None:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class OverlayConfig:
    """Display-only options; never used for inference or attribution."""

    boxes: bool = True
    skeletons: bool = True
    track_ids: bool = True
    labels: bool = True
    activity: bool = True
    trails: bool = False
    hud: bool = True
    line_thickness: int = 2
    text_scale: float = 1.0
    panel_alpha: float = 0.82
    trail_points: int = 30
    trail_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "OverlayConfig":
        return cls(
            boxes=_env_bool("MONITOR_OVERLAY_BOXES", True),
            skeletons=_env_bool("MONITOR_OVERLAY_SKELETONS", True),
            track_ids=_env_bool("MONITOR_OVERLAY_TRACK_IDS", True),
            labels=_env_bool("MONITOR_OVERLAY_LABELS", True),
            activity=_env_bool("MONITOR_OVERLAY_ACTIVITY", True),
            trails=_env_bool("MONITOR_OVERLAY_TRAILS", False),
            hud=_env_bool("MONITOR_OVERLAY_HUD", True),
            line_thickness=min(8, _env_int("MONITOR_OVERLAY_LINE_THICKNESS", 2)),
            text_scale=_env_float("MONITOR_OVERLAY_TEXT_SCALE", 1, minimum=0.5, maximum=2),
            panel_alpha=_env_float("MONITOR_OVERLAY_PANEL_ALPHA", 0.82, maximum=1),
            trail_points=min(300, _env_int("MONITOR_OVERLAY_TRAIL_POINTS", 30, minimum=2)),
            trail_seconds=_env_float("MONITOR_OVERLAY_TRAIL_SECONDS", 2, minimum=0.1, maximum=30),
        )


@dataclass(frozen=True)
class AppConfig:
    """Values that affect model loading, scoring, tracking, and reporting."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    enrollments_dir: Path = PROJECT_ROOT / "enrollments"
    input_dir: Path = PROJECT_ROOT / "input"
    output_dir: Path = PROJECT_ROOT / "output"
    log_dir: Path = PROJECT_ROOT / "output" / "logs"
    event_dir: Path = PROJECT_ROOT / "output" / "events"

    allow_model_downloads: bool = True
    identity_enabled: bool = True
    expression_enabled: bool = True
    allow_unchecked_enrollment: bool = False
    model_download_timeout_seconds: float = 60.0
    model_download_attempts: int = 3

    expression_interval_frames: int = 5
    expression_smoothing_seconds: float = 0.8
    identity_interval_frames: int = 8
    stale_track_frames: int = 90
    identity_consensus_window: int = 5
    identity_required_matches: int = 3
    identity_ttl_frames: int = 90
    identity_min_face_size: int = 64
    identity_min_face_confidence: float = 0.85
    identity_cosine_threshold: float = 0.363
    identity_min_score_margin: float = 0.03
    min_concern_expression_confidence: float = 0.65
    min_expression_display_confidence: float = 0.45

    pose_confidence: float = 0.30
    pose_detection_confidence: float = 0.10
    pose_iou: float = 0.45
    min_keypoint_confidence: float = 0.30
    overlays: OverlayConfig = field(default_factory=OverlayConfig)

    activity_model: str = "none"
    activity_checkpoint_path: Path = PROJECT_ROOT / "data" / "activity_models" / "mlp.pt"
    activity_sequence_length: int = 16
    activity_confidence_threshold: float = 0.50
    activity_inference_interval: int = 5
    activity_smoothing_window: int = 5

    debug_timing: bool = False
    event_logging_enabled: bool = True
    log_level: str = "INFO"
    demo_high_review_names: str = ""
    recording_retention_days: int = 30
    identity_retention_days: int = 365
    log_retention_days: int = 30

    @property
    def face_detector_model_path(self) -> Path:
        return self.data_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def face_recognizer_model_path(self) -> Path:
        return self.data_dir / "face_recognition_sface_2021dec.onnx"

    @property
    def pose_model_path(self) -> Path:
        return self.data_dir / "yolov8n-pose.pt"

    @property
    def emotion_face_model_path(self) -> Path:
        return FACE_DETECTOR_PATH

    @property
    def emotion_classifier_model_path(self) -> Path:
        return self.data_dir / "enet_b2_8.onnx"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration with safe environment-variable overrides."""
        root = _runtime_root()
        return cls(
            overlays=OverlayConfig.from_env(),
            recording_retention_days=_env_int("MONITOR_RECORDING_RETENTION_DAYS", 30),
            identity_retention_days=_env_int("MONITOR_IDENTITY_RETENTION_DAYS", 365),
            log_retention_days=_env_int("MONITOR_LOG_RETENTION_DAYS", 30),
            project_root=root,
            data_dir=root / "data",
            enrollments_dir=root / "enrollments",
            input_dir=root / "input",
            output_dir=root / "output",
            log_dir=root / "output" / "logs",
            event_dir=root / "output" / "events",
            allow_model_downloads=_env_bool("MONITOR_ALLOW_MODEL_DOWNLOADS", True),
            identity_enabled=_env_bool("MONITOR_IDENTITY_ENABLED", True),
            expression_enabled=_env_bool("MONITOR_EXPRESSION_ENABLED", True),
            model_download_timeout_seconds=_env_float(
                "MONITOR_MODEL_DOWNLOAD_TIMEOUT", 60.0, minimum=5.0
            ),
            model_download_attempts=_env_int("MONITOR_MODEL_DOWNLOAD_ATTEMPTS", 3),
            expression_interval_frames=_env_int("MONITOR_EXPRESSION_INTERVAL", 5),
            expression_smoothing_seconds=_env_float(
                "MONITOR_EXPRESSION_SMOOTHING_SECONDS", 0.8, minimum=0.1
            ),
            identity_interval_frames=_env_int("MONITOR_IDENTITY_INTERVAL", 8),
            stale_track_frames=_env_int("MONITOR_STALE_TRACK_FRAMES", 90),
            identity_consensus_window=_env_int("MONITOR_IDENTITY_WINDOW", 5),
            identity_required_matches=_env_int("MONITOR_IDENTITY_MATCHES", 3),
            identity_ttl_frames=_env_int("MONITOR_IDENTITY_TTL_FRAMES", 90),
            identity_min_face_size=_env_int("MONITOR_MIN_FACE_SIZE", 64),
            identity_min_face_confidence=_env_float(
                "MONITOR_MIN_FACE_CONFIDENCE", 0.85, maximum=1.0
            ),
            identity_cosine_threshold=_env_float("MONITOR_IDENTITY_THRESHOLD", 0.363, maximum=1.0),
            identity_min_score_margin=_env_float("MONITOR_IDENTITY_MARGIN", 0.03, maximum=1.0),
            min_concern_expression_confidence=_env_float(
                "MONITOR_EXPRESSION_CONFIDENCE", 0.65, maximum=1.0
            ),
            min_expression_display_confidence=_env_float(
                "MONITOR_EXPRESSION_DISPLAY_CONFIDENCE", 0.45, maximum=1.0
            ),
            pose_confidence=_env_float("MONITOR_POSE_CONFIDENCE", 0.30, maximum=1.0),
            pose_detection_confidence=_env_float(
                "MONITOR_POSE_DETECTION_CONFIDENCE", 0.10, minimum=0.01, maximum=0.10
            ),
            pose_iou=_env_float("MONITOR_POSE_IOU", 0.45, maximum=1.0),
            min_keypoint_confidence=_env_float("MONITOR_KEYPOINT_CONFIDENCE", 0.30, maximum=1.0),
            activity_model=_env_choice(
                "MONITOR_ACTIVITY_MODEL",
                "none",
                {"none", "mlp"},
            ),
            activity_checkpoint_path=_env_path(
                "MONITOR_ACTIVITY_CHECKPOINT",
                root / "data" / "activity_models" / "mlp.pt",
                root=root,
            ),
            activity_sequence_length=_env_int(
                "MONITOR_ACTIVITY_SEQUENCE_LENGTH",
                16,
            ),
            activity_confidence_threshold=_env_float(
                "MONITOR_ACTIVITY_CONFIDENCE",
                0.50,
                maximum=1.0,
            ),
            activity_inference_interval=_env_int(
                "MONITOR_ACTIVITY_INTERVAL",
                5,
            ),
            activity_smoothing_window=_env_int(
                "MONITOR_ACTIVITY_SMOOTHING_WINDOW",
                5,
            ),
            debug_timing=_env_bool("MONITOR_DEBUG_TIMING", False),
            event_logging_enabled=_env_bool("MONITOR_EVENT_LOGGING", True),
            log_level=_env_choice(
                "MONITOR_LOG_LEVEL",
                "info",
                {"debug", "info", "warning", "error", "critical", "notset"},
            ).upper(),
            demo_high_review_names=os.environ.get("DEMO_HIGH_REVIEW_NAMES", ""),
        )


DEFAULT_CONFIG = AppConfig.from_env()
