from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose import PoseAnalyzer, PoseResult, _clamp_box, TRACKER_CONFIG_PATH


class PoseAnalyzerOverlayTest(unittest.TestCase):
    def test_analyze_accepts_list_and_iterator_results(self) -> None:
        result = SimpleNamespace(
            boxes=SimpleNamespace(),
            keypoints=SimpleNamespace(xy=np.array([[[50.0, 25.0]]]), conf=np.array([[0.9]])),
        )

        class Boxes:
            id = np.array([7])
            conf = np.array([0.9])
            xyxy = np.array([[-10, 5, 120, 90]])

            def __len__(self):
                return 1

        result.boxes = Boxes()
        for results in ([result], iter([result])):
            with self.subTest(result_type=type(results).__name__):
                analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
                analyzer.model = Mock()
                analyzer.model.track.return_value = results
                analyzer.detection_confidence = 0.1
                analyzer.confidence = 0.3
                analyzer.iou = 0.45
                analyzer.min_keypoint_confidence = 0.3
                self.assertEqual(
                    analyzer.analyze(np.zeros((80, 100, 3), np.uint8)),
                    [PoseResult(7, (0, 5, 100, 80), {0: (0.5, 0.3125)})],
                )
                self.assertFalse(analyzer.model.track.call_args.kwargs["stream"])
                self.assertTrue(analyzer.model.track.call_args.kwargs["persist"])
                self.assertEqual(analyzer.model.track.call_args.kwargs["conf"], 0.1)
                self.assertEqual(
                    analyzer.model.track.call_args.kwargs["tracker"], str(TRACKER_CONFIG_PATH)
                )
        result.boxes.conf = np.array([0.15])
        analyzer.model.track.return_value = [result]
        self.assertEqual(analyzer.analyze(np.zeros((80, 100, 3), np.uint8)), [])

    def test_analyze_accepts_empty_result_iterables(self) -> None:
        for results in ([], iter([])):
            analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
            analyzer.model = Mock()
            analyzer.model.track.return_value = results
            analyzer.detection_confidence = 0.1
            analyzer.confidence = 0.3
            analyzer.iou = 0.45
            self.assertEqual(analyzer.analyze(np.zeros((80, 100, 3), np.uint8)), [])

    def test_person_box_is_clamped_to_frame_boundaries(self) -> None:
        self.assertEqual(_clamp_box((-10, 5, 120, 90), 100, 80), (0, 5, 100, 80))

    def test_landmarks_are_normalized_and_low_confidence_points_are_ignored(self) -> None:
        landmarks = PoseAnalyzer.landmarks_from_keypoints(
            np.array([[50.0, 25.0], [190.0, 90.0]], dtype=np.float32),
            np.array([0.90, 0.10], dtype=np.float32),
            frame_width=200,
            frame_height=100,
        )

        self.assertEqual(landmarks, {0: (0.25, 0.25)})


if __name__ == "__main__":
    unittest.main()
