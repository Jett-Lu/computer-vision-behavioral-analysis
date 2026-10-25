"""Generate synthetic overlay examples and measure rendering only (no AI models)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import AppConfig, OverlayConfig
from data_policy import atomic_output, ensure_private_directory
from detection import MonitoringProcessor
from events import EventRecorder, TrackSnapshot
from media_export import _export_image, _export_video
from overlays import OverlayRenderer, RenderPerson
from pose import PoseResult


class SyntheticPoseAnalyzer:
    """Scripted image coordinates, never model predictions or real people."""

    def __init__(self) -> None:
        self.index = 0

    def analyze(self, frame: np.ndarray) -> list[PoseResult]:
        height, width = frame.shape[:2]
        poses = []
        for index, center in enumerate((0.25, 0.70)):
            center += 0.04 * np.sin(self.index / 9 + index)
            points = {
                0: (center, 0.30),
                5: (center - 0.04, 0.39),
                6: (center + 0.04, 0.39),
                7: (center - 0.07, 0.52),
                8: (center + 0.08, 0.30),
                9: (center - 0.09, 0.62),
                10: (center + 0.10, 0.20),
                11: (center - 0.03, 0.62),
                12: (center + 0.03, 0.62),
                13: (center - 0.04, 0.77),
                14: (center + 0.05, 0.77),
                15: (center - 0.06, 0.91),
                16: (center + 0.07, 0.91),
            }
            poses.append(
                PoseResult(
                    7 + index,
                    (
                        int((center - 0.12) * width),
                        int(0.18 * height),
                        int((center + 0.13) * width),
                        int(0.93 * height),
                    ),
                    points,
                )
            )
        self.index += 1
        return poses

    def reset_tracking(self) -> None:
        self.index = 0


def background(width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.full((height, width, 3), (43, 35, 27), np.uint8)
    for x in range(0, width, 60):
        cv2.line(frame, (x, 0), (x, height - 1), (51, 43, 35), 1)
    for y in range(0, height, 60):
        cv2.line(frame, (0, y), (width - 1, y), (51, 43, 35), 1)
    cv2.putText(
        frame,
        "SYNTHETIC FIXTURE / NO REAL PEOPLE",
        (20, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return frame


def create_samples(directory: Path, frame_count: int = 60) -> dict[str, object]:
    ensure_private_directory(directory)
    config = replace(
        AppConfig(),
        identity_enabled=False,
        expression_enabled=False,
        allow_model_downloads=False,
        activity_model="none",
        log_dir=directory / "logs",
        overlays=OverlayConfig(trails=True),
    )
    frame = background()
    with atomic_output(directory / "sample.png") as temporary:
        if not cv2.imwrite(str(temporary), frame):
            raise RuntimeError("Failed to write synthetic image")
    with atomic_output(directory / "sample.mp4") as temporary:
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter.fourcc(*"mp4v"), 30, (960, 540))
        try:
            if not writer.isOpened():
                raise RuntimeError("MP4 codec unavailable")
            for _ in range(frame_count):
                writer.write(frame)
        finally:
            writer.release()
    processor = MonitoringProcessor(
        config, pose_analyzer=SyntheticPoseAnalyzer(), event_recorder=EventRecorder(None)
    )
    try:
        _export_image(directory / "sample.png", directory / "sample_annotated.png", processor)
        processor.reset_tracking(source="synthetic video")
        _export_video(directory / "sample.mp4", directory / "sample_annotated.mp4", processor)
    finally:
        processor.close()
    return {"frames": frame_count, "width": 960, "height": 540, "synthetic": True}


def benchmark(iterations: int = 100) -> dict[str, object]:
    frame = background(1280, 720)
    poses = SyntheticPoseAnalyzer().analyze(frame)
    people = [
        RenderPerson(
            p,
            TrackSnapshot(i, p.track_id, "Person", None, False, "CLEAR", 0, 0, None, 0),
            tuple((0.2 + j * 0.002, 0.8) for j in range(30)),
        )
        for i, p in enumerate(poses)
    ]
    rows = {}
    for name, settings in (
        (
            "copy_only",
            OverlayConfig(
                boxes=False,
                skeletons=False,
                track_ids=False,
                labels=False,
                activity=False,
                trails=False,
                hud=False,
            ),
        ),
        ("all_overlays", OverlayConfig(trails=True)),
    ):
        renderer = OverlayRenderer(settings, identity_enabled=False, expression_enabled=False)
        times = []
        for i in range(iterations + 10):
            start = time.perf_counter()
            renderer.draw(frame.copy(), people, fps=30, recent_events=())
            elapsed = (time.perf_counter() - start) * 1000
            if i >= 10:
                times.append(elapsed)
        rows[name] = {
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "resolution": "1280x720",
        "people": 2,
        "iterations": iterations,
        "warmup": 10,
        "includes_frame_copy": True,
        "excludes_inference_capture_encoding": True,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/overlay-samples"))
    args = parser.parse_args()
    result = create_samples(args.output)
    result["benchmark"] = benchmark()
    with atomic_output(args.output / "measurements.json") as temporary:
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
