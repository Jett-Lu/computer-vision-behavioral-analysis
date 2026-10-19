"""Export annotated monitoring results for photos and recorded videos."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2

from camera import CaptureClock
from config import DEFAULT_CONFIG, AppConfig
from logging_setup import get_logger
from data_policy import ensure_private_directory, atomic_output, _safe_path, safe_file_path

DEFAULT_INPUT_DIR = DEFAULT_CONFIG.input_dir
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG.output_dir
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
logger = get_logger("export")


@dataclass(frozen=True)
class ExportResult:
    """Describe one successfully generated annotated media file."""

    input_path: Path
    output_path: Path
    media_type: str
    frames_processed: int


def discover_media_files(input_path: Path) -> list[Path]:
    """Return supported input files from one file path or one directory."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not input_path.exists():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return []
        ensure_private_directory(input_path)
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def annotated_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build a clear output filename for a photo or video."""
    suffix = input_path.suffix.lower() if input_path.suffix.lower() in IMAGE_EXTENSIONS else ".mp4"
    name = input_path.stem if input_path.suffix.lower() in IMAGE_EXTENSIONS else input_path.name
    return output_dir / f"{name}_annotated{suffix}"


def export_media(
    input_path: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: AppConfig | None = None,
    *,
    fail_on_error: bool = False,
) -> list[ExportResult]:
    """Process supported media inputs and return the successfully generated files."""
    input_path = input_path.absolute()
    output_dir = _safe_path(output_dir)
    ensure_private_directory(output_dir)
    media_files = discover_media_files(input_path)
    if not media_files:
        if fail_on_error:
            raise RuntimeError(f"No supported photos or videos found in {input_path}")
        print(f"No supported photos or videos found in {input_path}")
        print(f"Add a file and run the command again. Output will appear in {output_dir}")
        return []

    input_files = set(media_files)
    if any(annotated_output_path(path, output_dir) in input_files for path in media_files):
        raise ValueError("Output paths overlap input files; choose a separate output directory.")

    results: list[ExportResult] = []
    processor = _create_processor(config)
    try:
        for media_path in media_files:
            output_path = safe_file_path(annotated_output_path(media_path, output_dir))
            print(f"\nProcessing {media_path.name}...")
            processor.reset_tracking(source=str(media_path))
            try:
                if media_path.suffix.lower() in IMAGE_EXTENSIONS:
                    result = _export_image(media_path, output_path, processor)
                else:
                    result = _export_video(media_path, output_path, processor)
            except Exception as exc:
                logger.exception("Failed to process media file: %s", media_path)
                print(f"Failed to process {media_path.name}: {exc}")
                continue
            results.append(result)
            print(f"Created {result.output_path}")
    finally:
        processor.close()

    print(f"\nCompleted {len(results)} of {len(media_files)} media files.")
    if fail_on_error and len(results) != len(media_files):
        raise RuntimeError(f"Failed to export {len(media_files) - len(results)} media file(s).")
    return results


def _export_image(input_path: Path, output_path: Path, processor: Any) -> ExportResult:
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError("OpenCV could not read the image.")

    annotated = image
    for frame_number in range(max(1, processor.config.identity_required_matches)):
        annotated = processor.process_frame(
            image,
            timestamp=frame_number / 30.0,
            force_identity=True,
        )

    with atomic_output(output_path) as temporary_path:
        if not cv2.imwrite(str(temporary_path), annotated):
            raise RuntimeError("OpenCV could not write the annotated image.")
    return ExportResult(input_path, output_path, "image", frames_processed=1)


def _export_video(input_path: Path, output_path: Path, processor: Any) -> ExportResult:
    safe_file_path(output_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the video.")

    writer: Any = None
    frames_processed = 0
    with ExitStack() as stack:
        stack.callback(capture.release)
        temporary_path = stack.enter_context(atomic_output(output_path))
        try:
            clock = CaptureClock(capture, recorded=True)
            fps = clock.fps
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise RuntimeError("The video does not report a valid frame size.")

            writer = cv2.VideoWriter(
                str(temporary_path),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not create the annotated MP4 output.")

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = clock.timestamp()
                writer.write(processor.process_frame(frame, timestamp=timestamp))
                frames_processed += 1
                if frames_processed % 30 == 0:
                    print(f"Processed {frames_processed} frames...")
        finally:
            if writer is not None:
                writer.release()

        if frames_processed == 0:
            raise RuntimeError("The video did not contain readable frames.")
    return ExportResult(input_path, output_path, "video", frames_processed)


def _create_processor(config: AppConfig | None = None) -> Any:
    from detection import MonitoringProcessor

    return MonitoringProcessor(config)
