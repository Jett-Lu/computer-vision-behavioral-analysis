from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DEFAULT_CONFIG
from detection import (
    MonitoringProcessor,
    PersonRuntime,
    _apply_demo_review_override,
    _discard_stale_tracks,
    _parse_demo_high_review_names,
    _source_for_logging,
    _track_key,
    run_detection,
)
from events import EventRecorder
from identity import IdentityConsensus
from pose import PoseResult
from review import CLEAR_COLOR, HIGH_COLOR, ReviewState


class DetectionAssociationTest(unittest.TestCase):
    def test_tracker_id_is_used_as_person_state_key(self) -> None:
        self.assertEqual(_track_key(42, 0), 42)
        self.assertEqual(_track_key(None, 0, frame_number=0), -10001)
        self.assertNotEqual(
            _track_key(None, 0, frame_number=0),
            _track_key(None, 0, frame_number=1),
        )

    def test_stale_person_state_and_identity_name_are_removed(self) -> None:
        person_states = {
            3: PersonRuntime(None, None, IdentityConsensus(), last_seen_frame=0),
            4: PersonRuntime(None, None, IdentityConsensus(), last_seen_frame=100),
        }

        stale = _discard_stale_tracks(person_states, frame_count=100)

        self.assertNotIn(3, person_states)
        self.assertIn(4, person_states)
        self.assertEqual(stale, [3])

    def test_demo_high_review_names_are_parsed_case_insensitively(self) -> None:
        self.assertEqual(
            _parse_demo_high_review_names("Alex Chen; Priya Shah,  "),
            {"alex chen", "priya shah"},
        )

    def test_demo_high_review_override_only_changes_named_identity(self) -> None:
        state = ReviewState("CLEAR", CLEAR_COLOR, 0.0, 0, 0.0, None)

        unchanged = _apply_demo_review_override(state, "Jordan Lee", {"alex chen"})
        overridden = _apply_demo_review_override(state, "Alex Chen", {"alex chen"})

        self.assertIs(unchanged, state)
        self.assertEqual(overridden.tier_label, "HIGH")
        self.assertEqual(overridden.color, HIGH_COLOR)
        self.assertEqual(overridden.recent_wave_count, 5)

    def test_camera_credentials_are_removed_from_logged_source(self) -> None:
        self.assertEqual(
            _source_for_logging("rtsp://user:secret@example.test:8554/live?token=abc"),
            "rtsp://example.test:8554/live",
        )
        self.assertEqual(_source_for_logging("input\\clip.mp4"), "input\\clip.mp4")


class FakePoseAnalyzer:
    def analyze(self, frame):
        return [
            PoseResult(
                track_id=7,
                box=(10, 10, 80, 90),
                landmarks={},
            )
        ]

    def draw_landmarks(self, frame, landmarks, color):
        return None


class UntrackedPoseAnalyzer(FakePoseAnalyzer):
    def analyze(self, frame):
        return [PoseResult(track_id=None, box=(10, 10, 80, 90), landmarks={})]


class RecordingEventRecorder:
    def advance_clock(self, timestamp, frame_number):
        return None

    def mark_unavailable(self, track_key, timestamp, frame_number):
        return None

    def __init__(self) -> None:
        self.recorded = []
        self.recent_messages = ()

    def record(self, snapshot, timestamp, frame_number):
        self.recorded.append(snapshot)

    def end_track(self, track_key, timestamp, frame_number):
        return None

    def reset_tracks(self, **context):
        return None

    def close(self):
        return None


class MonitoringProcessorIntegrationTest(unittest.TestCase):
    def test_close_releases_resources_when_activity_metrics_fail(self) -> None:
        from unittest.mock import PropertyMock

        activity = Mock()
        type(activity).average_inference_latency_ms = PropertyMock(
            side_effect=RuntimeError("metrics unavailable")
        )
        processor = MonitoringProcessor(
            replace(DEFAULT_CONFIG, activity_model="none"),
            pose_analyzer=FakePoseAnalyzer(),
            emotion_analyzer=Mock(),
            identity_matcher=None,
            activity_recognizer=activity,
            event_recorder=Mock(),
        )
        with self.assertRaisesRegex(RuntimeError, "metrics unavailable"):
            processor.close()
        processor.emotion_analyzer.close.assert_called_once()
        processor.event_recorder.close.assert_called_once()

    def test_anonymous_label_cannot_trigger_demo_override(self) -> None:
        processor = MonitoringProcessor(
            replace(DEFAULT_CONFIG, demo_high_review_names="Person", activity_model="none"),
            pose_analyzer=FakePoseAnalyzer(),
            emotion_analyzer=None,
            identity_matcher=None,
            event_recorder=EventRecorder(None),
        )
        try:
            processor.process_frame(np.zeros((100, 100, 3), np.uint8), timestamp=0.0)
            snapshot = processor.last_snapshots[0]
            self.assertEqual(snapshot.tier_label, "CLEAR")
            self.assertFalse(snapshot.demo_override)
        finally:
            processor.close()

    def test_unavailable_capture_reports_failure(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("detection.open_capture", return_value=capture):
            self.assertIs(run_detection(0), False)
        capture.release.assert_called_once()

    def test_recorded_detection_passes_media_time_to_processor(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
        positions = iter([0.0, 100.0])
        capture.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_FPS else next(positions)
        processor = Mock()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recording.mp4"
            source.touch()
            with (
                patch("detection.open_capture", return_value=capture),
                patch("detection.MonitoringProcessor", return_value=processor),
                patch("detection.cv2.imshow"),
                patch("detection.cv2.waitKey", return_value=-1),
                patch("detection.cv2.destroyAllWindows"),
            ):
                run_detection(str(source))
        self.assertEqual(
            [call.kwargs["timestamp"] for call in processor.process_frame.call_args_list],
            [0.0, 0.1],
        )
        capture.release.assert_called_once()
        processor.close.assert_called_once()

    def test_pipeline_records_the_actual_wave_signal(self) -> None:
        from gesture import WaveAlertState

        recorder = RecordingEventRecorder()
        processor = MonitoringProcessor(
            replace(DEFAULT_CONFIG, activity_model="none"),
            pose_analyzer=FakePoseAnalyzer(),
            emotion_analyzer=None,
            identity_matcher=None,
            event_recorder=recorder,
        )
        wave_monitor = Mock()
        wave_monitor.update.return_value = WaveAlertState(1, True)
        processor.wave_monitor_type = lambda: wave_monitor
        try:
            processor.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0.0)
        finally:
            processor.close()
        self.assertTrue(recorder.recorded[0].wave_detected)

    def test_frame_pipeline_runs_with_optional_models_disabled(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            allow_model_downloads=False,
            event_logging_enabled=False,
            expression_interval_frames=1,
        )
        processor = MonitoringProcessor(
            config,
            pose_analyzer=FakePoseAnalyzer(),
            emotion_analyzer=None,
            identity_matcher=None,
            known_identities=[],
            event_recorder=EventRecorder(None),
        )
        try:
            annotated = processor.process_frame(
                np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(annotated.shape, (100, 100, 3))
        self.assertEqual(len(processor.last_snapshots), 1)
        self.assertEqual(processor.last_snapshots[0].track_id, 7)
        self.assertEqual(processor.last_snapshots[0].identity_name, "Person")

    def test_placeholder_tracks_are_not_written_to_event_reports(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            allow_model_downloads=False,
            event_logging_enabled=False,
            expression_interval_frames=1,
        )
        recorder = RecordingEventRecorder()
        processor = MonitoringProcessor(
            config,
            pose_analyzer=UntrackedPoseAnalyzer(),
            emotion_analyzer=None,
            identity_matcher=None,
            known_identities=[],
            event_recorder=recorder,
        )
        try:
            processor.process_frame(
                np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(recorder.recorded, [])
        self.assertEqual(len(processor.last_snapshots), 1)


if __name__ == "__main__":
    unittest.main()
