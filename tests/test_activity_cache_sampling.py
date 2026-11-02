"""Cache provenance and shared offline/live source-time sampling regressions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_activity_prediction_expiry import BOX, FRAME_SHAPE, LANDMARKS, WALKING, make_recognizer

from activity_recognition.dataset import ActivitySample
from activity_recognition.preprocessing import cache_activity_samples, sample_video_frames
from activity_recognition.train import cache_backbone_features, train_classifier


class CacheSamplingTest(unittest.TestCase):
    def test_offline_sampling_uses_presentation_times_for_variable_rate_video(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30.0 if prop == cv2.CAP_PROP_FPS else positions.pop(0)
        )
        positions = [0.0, 40.0, 150.0, 300.0]
        capture.read.side_effect = [
            (True, np.full((2, 2, 3), value, np.uint8)) for value in range(4)
        ] + [(False, None)]
        with patch("activity_recognition.preprocessing.cv2.VideoCapture", return_value=capture):
            frames = sample_video_frames(Path("variable.mp4"), 4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [0, 1, 2, 3])
        capture.release.assert_called_once()

    def test_non_object_feature_metadata_is_regenerated(self):
        class Extractor(torch.nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        np.savez(self.sample.cache_path, crops=np.zeros((2, 8, 8, 3), np.uint8))
        with patch(
            "activity_recognition.train.build_mobilenet_extractor", return_value=Extractor()
        ):
            cache_backbone_features([self.sample], self.root, "cnn", torch.device("cpu"))
            (self.root / "cnn" / "clip.json").write_text("[]", encoding="utf-8")
            cache_backbone_features([self.sample], self.root, "cnn", torch.device("cpu"))
        self.assertEqual(np.load(self.root / "cnn" / "clip.npy").shape, (3,))

    def setUp(self):
        contract = patch(
            "activity_recognition.preprocessing.pose_extraction_provenance",
            return_value={"fixture": "mock-pose"},
        )
        contract.start()
        self.addCleanup(contract.stop)
        weights = patch(
            "activity_recognition.train.backbone_weight_fingerprint", return_value="mock-backbone"
        )
        weights.start()
        self.addCleanup(weights.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "clip.avi"
        source.write_bytes(b"source video")
        self.sample = ActivitySample(
            "clip", str(source), str(self.root / "clip.npz"), "walking", 0, "train", "walk"
        )

    def test_pose_cache_reuses_valid_but_regenerates_changed_frames_source_and_corruption(self):
        analyzer = Mock()
        analyzer.analyze.return_value = []

        def frames(path, count, **kwargs):
            return [np.zeros((8, 8, 3), np.uint8)] * count

        with patch("activity_recognition.preprocessing.sample_video_frames", side_effect=frames):
            self.assertEqual(
                cache_activity_samples([self.sample], frames_per_sample=2, pose_analyzer=analyzer)[
                    "completed"
                ],
                1,
            )
            self.assertEqual(
                cache_activity_samples([self.sample], frames_per_sample=2)["skipped"], 1
            )
            self.assertEqual(
                cache_activity_samples([self.sample], frames_per_sample=3, pose_analyzer=analyzer)[
                    "completed"
                ],
                1,
            )
            Path(self.sample.video_path).write_bytes(b"changed source")
            self.assertEqual(
                cache_activity_samples([self.sample], frames_per_sample=3, pose_analyzer=analyzer)[
                    "completed"
                ],
                1,
            )
            Path(self.sample.cache_path).write_bytes(b"broken zip")
            self.assertEqual(
                cache_activity_samples([self.sample], frames_per_sample=3, pose_analyzer=analyzer)[
                    "completed"
                ],
                1,
            )
        with np.load(self.sample.cache_path) as data:
            self.assertEqual(data["pose"].shape, (3, 17, 3))

    def test_backbone_cache_reuses_and_invalidates_for_each_split(self):
        class Extractor(torch.nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        for split in ("train", "validation", "test"):
            sample = ActivitySample(
                split,
                self.sample.video_path,
                str(self.root / f"{split}.npz"),
                "walking",
                0,
                split,
                "walk",
            )
            np.savez(sample.cache_path, crops=np.zeros((2, 8, 8, 3), np.uint8))
            with patch(
                "activity_recognition.train.build_mobilenet_extractor", return_value=Extractor()
            ) as build:
                cache_backbone_features([sample], self.root, "cnn", torch.device("cpu"))
                output = self.root / "cnn" / f"{split}.npy"
                first = np.load(output)
                cache_backbone_features([sample], self.root, "cnn", torch.device("cpu"))
                self.assertEqual(build.call_count, 1)
                np.savez(sample.cache_path, crops=np.full((2, 8, 8, 3), 255, np.uint8))
                cache_backbone_features([sample], self.root, "cnn", torch.device("cpu"))
                self.assertEqual(build.call_count, 2)
                self.assertFalse(np.array_equal(first, np.load(output)))
                output.write_bytes(b"broken npy")
                cache_backbone_features([sample], self.root, "cnn", torch.device("cpu"))
                self.assertEqual(build.call_count, 3)

    def test_offline_sampling_has_fixed_duration_and_duplicates_short_clips(self):
        class Capture:
            def __init__(self, count):
                self.count, self.index = count, 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 20.0

            def read(self):
                if self.index >= self.count:
                    return False, None
                value = self.index
                self.index += 1
                return True, np.full((2, 2, 3), value, np.uint8)

            def release(self):
                pass

        long = Capture(200)
        with patch("activity_recognition.preprocessing.cv2.VideoCapture", return_value=long):
            result = sample_video_frames(Path("long.avi"), 4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in result], [0, 2, 4, 6])
        self.assertLessEqual(long.index, 7)
        with patch("activity_recognition.preprocessing.cv2.VideoCapture", return_value=Capture(2)):
            result = sample_video_frames(Path("short.avi"), 4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in result], [0, 1, 1, 1])

    def test_training_rejects_mlp_frame_count_mismatch_before_building_model(self):
        with self.assertRaisesRegex(ValueError, "frame count"):
            train_classifier(
                "mlp",
                np.zeros((2, 102), np.float32),
                np.array([0, 1]),
                np.zeros((2, 102), np.float32),
                np.array([0, 1]),
                self.root / "bad.pt",
                epochs=1,
                patience=1,
                batch_size=2,
                learning_rate=0.001,
                seed=1,
                device=torch.device("cpu"),
                manifest_metadata={"frames_per_sample": 3},
            )
        self.assertFalse((self.root / "bad.pt").exists())

    def test_live_source_time_sampling_and_gap_reset_are_per_track(self):
        recognizer = make_recognizer(sequence_length=3)
        recognizer.sampling_interval_seconds = 0.1
        with patch.object(recognizer, "_predict_probabilities", return_value=WALKING) as predict:
            for frame, timestamp in enumerate((0.0, 0.025, 0.05, 0.075, 0.1, 0.15)):
                self.assertIsNone(
                    recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, frame, timestamp=timestamp)
                )
            self.assertEqual(
                recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 6, timestamp=0.2).label, "walking"
            )
            self.assertEqual(predict.call_count, 1)
            self.assertIsNone(recognizer.update(2, LANDMARKS, BOX, FRAME_SHAPE, 6, timestamp=0.2))
            self.assertIsNone(recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 7, timestamp=1.0))
            self.assertEqual(len(recognizer._track_states[1].pose_history), 1)
            self.assertEqual(len(recognizer._track_states[2].pose_history), 1)


if __name__ == "__main__":
    unittest.main()
