"""Reproducible, atomic model provisioning for first-run setup."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import time

from config import AppConfig
from data_policy import atomic_output, ensure_private_directory, safe_file_path
from logging_setup import get_logger


logger = get_logger("models")


class ModelUnavailableError(RuntimeError):
    """Raised when a required local model cannot be verified or downloaded."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    url: str
    sha256: str


FACE_DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_RECOGNIZER_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
POSE_MODEL_SHA256 = "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"
EMOTION_MODEL_SHA256 = "180a9d4845b59393de4511598a0d1d34b705034691ea32959ce5009db7cf52b7"
OPENCV_ZOO_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
EMOTIEFFLIB_REVISION = "95d1a227cac48e31f9557acb96afed455695c09b"


def identity_model_specs(config: AppConfig) -> tuple[ModelSpec, ModelSpec]:
    return (
        ModelSpec(
            "YuNet face detector",
            config.face_detector_model_path,
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_ZOO_REVISION}/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            FACE_DETECTOR_SHA256,
        ),
        ModelSpec(
            "SFace recognizer",
            config.face_recognizer_model_path,
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_ZOO_REVISION}/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            FACE_RECOGNIZER_SHA256,
        ),
    )


def pose_model_spec(config: AppConfig) -> ModelSpec:
    return ModelSpec(
        "YOLO pose model",
        config.pose_model_path,
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt",
        POSE_MODEL_SHA256,
    )


def emotion_model_spec(config: AppConfig) -> ModelSpec:
    return ModelSpec(
        "EmotiEffLib expression model",
        config.emotion_classifier_model_path,
        f"https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/"
        f"{EMOTIEFFLIB_REVISION}/models/affectnet_emotions/onnx/enet_b2_8.onnx",
        EMOTION_MODEL_SHA256,
    )


def ensure_models(specs: tuple[ModelSpec, ...], config: AppConfig) -> None:
    for spec in specs:
        ensure_model(spec, config)


def ensure_model(spec: ModelSpec, config: AppConfig) -> Path:
    """Return a verified model path, downloading atomically when permitted."""
    safe_file_path(spec.path)
    verification_error: OSError | None = None
    if spec.path.exists():
        try:
            if file_sha256(spec.path) == spec.sha256.casefold():
                logger.info("Model ready: %s", spec.name)
                return spec.path
        except OSError as exc:
            verification_error = exc

    if not config.allow_model_downloads:
        raise ModelUnavailableError(
            f"{spec.name} is missing or invalid at {spec.path}. Model downloads are disabled."
        ) from verification_error

    try:
        ensure_private_directory(spec.path.parent)
    except OSError as exc:
        raise ModelUnavailableError(
            f"Could not create the model directory for {spec.name}: {spec.path.parent}"
        ) from exc
    last_error: Exception | None = None
    for attempt in range(1, config.model_download_attempts + 1):
        try:
            logger.info(
                "Downloading %s (attempt %s/%s)",
                spec.name,
                attempt,
                config.model_download_attempts,
            )
            with atomic_output(spec.path) as temporary_path:
                request = Request(spec.url, headers={"User-Agent": "monitoring-poc/1.0"})
                with urlopen(request, timeout=config.model_download_timeout_seconds) as response:
                    with temporary_path.open("wb") as destination:
                        shutil.copyfileobj(response, destination)

                actual_hash = file_sha256(temporary_path)
                if actual_hash != spec.sha256:
                    raise ModelUnavailableError(
                        f"Checksum mismatch for {spec.name}: expected {spec.sha256}, "
                        f"received {actual_hash}."
                    )

            logger.info("Model installed: %s", spec.path)
            return spec.path
        except Exception as exc:
            last_error = exc
            logger.warning("Could not provision %s: %s", spec.name, exc)
            if attempt < config.model_download_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))

    raise ModelUnavailableError(
        f"Could not prepare {spec.name} at {spec.path}. Last error: {last_error}"
    ) from last_error


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
