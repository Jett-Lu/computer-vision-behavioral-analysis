from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_export import annotated_output_path, discover_media_files, export_media


class MediaExportHelpersTest(unittest.TestCase):
    def test_overlap_check_compares_canonical_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "child").mkdir()
            (root / "photo.png").write_bytes(b"original")
            protected = root / "photo_annotated.png"
            protected.write_bytes(b"keep")
            with patch("media_export._create_processor"):
                with self.assertRaisesRegex(ValueError, "input"):
                    export_media(root / "child" / "..", root)
            self.assertEqual(protected.read_bytes(), b"keep")

    def test_export_refuses_to_overwrite_another_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "photo.png").write_bytes(b"original")
            protected = root / "photo_annotated.png"
            protected.write_bytes(b"second input")
            with patch("media_export._create_processor"):
                with self.assertRaisesRegex(ValueError, "input"):
                    export_media(root, root)
            self.assertEqual(protected.read_bytes(), b"second input")

    def test_strict_export_fails_when_no_input_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "No supported"):
                export_media(root / "missing.mp4", root / "output", fail_on_error=True)

    def test_discover_media_files_returns_supported_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "notes.txt").touch()
            (folder / "clip.mp4").touch()
            (folder / "photo.jpg").touch()

            self.assertEqual(
                discover_media_files(folder),
                [folder / "clip.mp4", folder / "photo.jpg"],
            )

    def test_missing_file_path_is_not_created_as_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_file = Path(temp_dir) / "missing.mp4"

            self.assertEqual(discover_media_files(missing_file), [])
            self.assertFalse(missing_file.exists())

    def test_annotated_output_path_keeps_image_format(self) -> None:
        self.assertEqual(
            annotated_output_path(Path("input/photo.png"), Path("output")),
            Path("output/photo_annotated.png"),
        )

    def test_annotated_output_path_converts_video_to_mp4(self) -> None:
        self.assertEqual(
            annotated_output_path(Path("input/clip.mov"), Path("output")),
            Path("output/clip.mov_annotated.mp4"),
        )

    def test_same_stem_video_formats_have_distinct_outputs(self) -> None:
        paths = {
            annotated_output_path(Path(f"input/clip.{extension}"), Path("output"))
            for extension in ("mov", "avi", "mp4", "mkv", "m4v")
        }
        self.assertEqual(len(paths), 5)

    def test_same_stem_videos_export_separately_with_media_timestamps(self) -> None:
        class CopyProcessor:
            def __init__(self) -> None:
                self.timestamps: list[list[float]] = []
                self.closed = False

            def reset_tracking(self, **context) -> None:
                self.timestamps.append([])

            def process_frame(self, frame, *, timestamp):
                self.timestamps[-1].append(timestamp)
                return frame.copy()

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            for extension, codec, color in (
                ("avi", "MJPG", (0, 0, 255)),
                ("mp4", "mp4v", (255, 0, 0)),
            ):
                writer = cv2.VideoWriter(
                    str(inputs / f"clip.{extension}"),
                    cv2.VideoWriter_fourcc(*codec),
                    10.0,
                    (64, 48),
                )
                try:
                    self.assertTrue(writer.isOpened())
                    for _ in range(3):
                        writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
                finally:
                    writer.release()
            processor = CopyProcessor()
            with patch("media_export._create_processor", return_value=processor):
                results = export_media(inputs, root / "output")
            self.assertEqual(len(results), 2)
            self.assertEqual(len({result.output_path for result in results}), 2)
            self.assertTrue(processor.closed)
            for timestamps in processor.timestamps:
                np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2], atol=1e-6)
            for result, dominant_channel in zip(results, (2, 0)):
                self.assertEqual(result.frames_processed, 3)
                capture = cv2.VideoCapture(str(result.output_path))
                try:
                    ok, image = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(int(image.mean(axis=(0, 1)).argmax()), dominant_channel)
                finally:
                    capture.release()


if __name__ == "__main__":
    unittest.main()
