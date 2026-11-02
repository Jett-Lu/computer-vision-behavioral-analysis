from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.dataset import (
    ActivitySample,
    build_hmdb51_metadata_fold,
    deterministic_validation_keys,
    load_manifest,
    write_manifest,
)
from activity_recognition.preprocessing import (
    cache_activity_samples,
    crop_person,
    normalize_pose,
)


class ActivityPreprocessingTest(unittest.TestCase):
    def setUp(self):
        contract = patch(
            "activity_recognition.preprocessing.pose_extraction_provenance",
            return_value={"fixture": "mock-pose"},
        )
        contract.start()
        self.addCleanup(contract.stop)

    def test_pose_is_normalized_relative_to_person_box(self) -> None:
        features = normalize_pose(
            {0: (0.30, 0.40), 1: (0.50, 0.70)},
            (20, 20, 60, 80),
            (100, 100, 3),
        )

        np.testing.assert_allclose(features[0], (0.25, 1 / 3, 1.0), atol=1e-6)
        np.testing.assert_allclose(features[1], (0.75, 5 / 6, 1.0), atol=1e-6)
        self.assertEqual(float(features[2, 2]), 0.0)

    def test_missing_pose_returns_an_all_zero_sequence(self) -> None:
        features = normalize_pose({}, None, (100, 100, 3))

        np.testing.assert_array_equal(features, np.zeros((17, 3), dtype=np.float32))

    def test_person_crop_has_stable_rgb_shape(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[:, :, 2] = 255

        crop = crop_person(frame, (20, 10, 80, 70), output_size=64)

        self.assertEqual(crop.shape, (64, 64, 3))
        self.assertEqual(tuple(crop[0, 0]), (255, 0, 0))

    def test_validation_split_is_deterministic_and_video_level(self) -> None:
        names = [f"clip_{index}.avi" for index in range(20)]

        first = deterministic_validation_keys(names, 0.2, seed=7)
        second = deterministic_validation_keys(reversed(names), 0.2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(first.issubset(set(names)))

    def test_manifest_rejects_duplicate_source_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "walk" / "clip.avi"
            samples = [
                ActivitySample(
                    key="train-key",
                    video_path=str(source),
                    cache_path=str(root / "cache" / "train.npz"),
                    label="walking",
                    label_index=0,
                    split="train",
                    hmdb51_class="walk",
                ),
                ActivitySample(
                    key="test-key",
                    video_path=str(source),
                    cache_path=str(root / "cache" / "test.npz"),
                    label="walking",
                    label_index=0,
                    split="test",
                    hmdb51_class="walk",
                ),
            ]
            manifest_path = root / "manifest.json"
            write_manifest(
                manifest_path,
                samples,
                dataset_root=root,
                annotations_root=root,
                fold=1,
                validation_fraction=0.2,
                seed=7,
                frames_per_sample=16,
            )

            with self.assertRaisesRegex(ValueError, "Duplicate activity source"):
                load_manifest(manifest_path)

    def test_metadata_rejects_parent_directory_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.csv"
            test_path = root / "test.csv"
            for path, file_name in (
                (training_path, "../outside.avi"),
                (test_path, "videos/walk/test.avi"),
            ):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("file_name", "label"))
                    writer.writeheader()
                    writer.writerow({"file_name": file_name, "label": "walk"})

            with self.assertRaisesRegex(ValueError, "unsafe video path"):
                build_hmdb51_metadata_fold(
                    root,
                    training_path,
                    test_path,
                    root / "cache",
                )

    def test_existing_sample_cache_does_not_load_pose_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cached.npz"
            source_path = Path(directory) / "clip.avi"
            source_path.write_bytes(b"source identity for mocked decoder")
            sample = ActivitySample(
                key="cached",
                video_path=str(source_path),
                cache_path=str(cache_path),
                label="walking",
                label_index=0,
                split="train",
                hmdb51_class="walk",
            )

            analyzer = Mock()
            analyzer.analyze.return_value = []
            frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 2
            with patch(
                "activity_recognition.preprocessing.sample_video_frames",
                return_value=frames,
            ):
                initial_counts = cache_activity_samples(
                    [sample], frames_per_sample=2, pose_analyzer=analyzer
                )
            self.assertEqual(initial_counts, {"completed": 1, "skipped": 0, "failed": 0})
            original_cache = cache_path.read_bytes()
            with patch("pose.PoseAnalyzer") as build_analyzer:
                counts = cache_activity_samples([sample], frames_per_sample=2)
            build_analyzer.assert_not_called()
            self.assertEqual(cache_path.read_bytes(), original_cache)

        self.assertEqual(counts, {"completed": 0, "skipped": 1, "failed": 0})


if __name__ == "__main__":
    unittest.main()
