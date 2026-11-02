"""Deterministic tracking/attribution regressions; no camera or model downloads."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import IdentityConsensus, IdentityMatch, KnownIdentity
from gesture import RightHandWaveMonitor
from events import EventRecorder, TrackSnapshot
from config import AppConfig


def state(**changes):
    return replace(TrackSnapshot(7, 7, "Alex", 0.8, True, "CLEAR", 0, 0, None, 0), **changes)


class IdentityAttributionTest(unittest.TestCase):
    def test_pending_votes_expire_during_missing_observations(self):
        consensus = IdentityConsensus(required_matches=3, ttl_frames=5)
        known = IdentityMatch("Alex", 0.8, True)
        consensus.observe(known, 0)
        consensus.observe(known, 1)
        consensus.observe(None, 10)
        self.assertFalse(consensus.observe(known, 11).confirmed)

    def test_rejected_match_clears_name_but_missing_observation_preserves_ttl(self):
        known = IdentityMatch("Alex", 0.8, True)
        consensus = IdentityConsensus(required_matches=2)
        consensus.observe(known, 0)
        self.assertTrue(consensus.observe(known, 1).confirmed)
        self.assertTrue(consensus.observe(None, 2).confirmed)
        self.assertFalse(consensus.observe(IdentityMatch("Unknown", 0.1, False), 3).confirmed)
        self.assertFalse(consensus.observe(known, 4).confirmed)
        self.assertTrue(consensus.observe(known, 5).confirmed)

    def test_contradiction_cannot_reactivate_old_votes(self):
        consensus = IdentityConsensus(required_matches=3)
        alex = IdentityMatch("Alex", 0.8, True)
        for frame in range(3):
            consensus.observe(alex, frame)
        self.assertFalse(consensus.observe(IdentityMatch("Jo", 0.8, True), 3).confirmed)
        self.assertFalse(consensus.observe(alex, 4).confirmed)


class ScaleInvariantWaveTest(unittest.TestCase):
    def test_jitter_and_interrupted_motion_do_not_count(self):
        for offsets in ((-0.05, 0.05, -0.05, 0.05), (-0.6, 0.6, None, -0.6, 0.6)):
            monitor = RightHandWaveMonitor()
            for i, offset in enumerate(offsets):
                points = (
                    {}
                    if offset is None
                    else {6: (0.5, 0.5), 5: (0.3, 0.5), 10: (0.5 + offset * 0.2, 0.3)}
                )
                result = monitor.update(points, i * 0.15, (1080, 1920, 3))
            self.assertEqual(result.recent_wave_count, 0)

    def test_long_observation_gaps_break_motion(self):
        monitor = RightHandWaveMonitor()
        for i, offset in enumerate((-0.6, 0.6, -0.6, 0.6)):
            result = monitor.update(
                {6: (0.5, 0.5), 5: (0.3, 0.5), 10: (0.5 + offset * 0.2, 0.3)}, i
            )
        self.assertEqual(result.recent_wave_count, 0)

    def test_equivalent_near_and_far_waves_count_once(self):
        for scale in (0.015, 0.04, 0.10, 0.25):
            monitor = RightHandWaveMonitor()
            for i, relative_x in enumerate((-0.6, 0.6, -0.6, 0.6)):
                points = {
                    6: (0.5, 0.5),
                    5: (0.5 - scale, 0.5),
                    10: (0.5 + relative_x * scale, 0.5 - scale),
                }
                result = monitor.update(points, i * 0.15)
            self.assertEqual(result.recent_wave_count, 1, scale)

    def test_missing_body_scale_cannot_generate_wave(self):
        monitor = RightHandWaveMonitor()
        for i, x in enumerate((0.3, 0.7, 0.3, 0.7)):
            result = monitor.update({6: (0.5, 0.5), 10: (x, 0.3)}, i * 0.15)
        self.assertEqual(result.recent_wave_count, 0)


class LifecycleTest(unittest.TestCase):
    def test_display_expression_changes_and_loss_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(
                state(display_expression_label="Happiness?", display_expression_confidence=0.4),
                0,
                0,
            )
            recorder.record(state(), 1, 1)
            recorder.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(events[0]["state"]["display_expression_label"], "Happiness?")
        self.assertEqual(events[1]["event_type"], "expression_cleared")
        self.assertIsNone(events[1]["state"]["display_expression_label"])

    def test_shutdown_uses_latest_frame_even_without_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = EventRecorder(path, identity_enabled=False, expression_enabled=False)
            recorder.record(state(expression_label="Neutral"), 0, 0)
            recorder.mark_unavailable(7, 1, 1)
            recorder.advance_clock(5, 50)
            recorder.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(events[-1]["frame_number"], 50)
        self.assertEqual(events[-1]["timestamp_seconds"], 5)
        self.assertFalse(events[-1]["state"]["visible"])
        for event in events:
            self.assertNotIn("identity_name", event)
            self.assertNotIn("identity_name", event["state"])
            self.assertNotIn("expression_label", event["state"])

    def test_losses_reset_and_shutdown_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(
                state(
                    activity_label="walking", activity_confidence=0.8, expression_label="Neutral"
                ),
                0,
                0,
            )
            recorder.record(
                state(identity_name="Unknown", identity_confirmed=False, identity_score=None), 1, 1
            )
            recorder.reset_tracks(source="second")
            recorder.record(state(identity_name="Person", identity_confirmed=False), 0, 0)
            recorder.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
        kinds = [event["event_type"] for event in events]
        for kind in ("identity_cleared", "expression_cleared", "activity_unavailable"):
            self.assertIn(kind, kinds)
        endings = [e for e in events if e["event_type"] == "track_ended"]
        self.assertEqual([e["details"]["reason"] for e in endings], ["source_reset", "shutdown"])
        self.assertNotEqual(events[0]["session_id"], events[-1]["session_id"])


class TrackerConfigurationTest(unittest.TestCase):
    def test_detection_candidates_reach_low_confidence_stage(self):
        self.assertLessEqual(AppConfig().pose_detection_confidence, 0.1)

    def test_bytetrack_recovers_low_score_and_short_occlusion_without_new_ids(self):
        from dependency_privacy import disable_ultralytics_telemetry

        disable_ultralytics_telemetry()
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        from types import SimpleNamespace
        from ultralytics.utils import YAML
        from pose import TRACKER_CONFIG_PATH

        tracker = BYTETracker(SimpleNamespace(**YAML.load(TRACKER_CONFIG_PATH)))

        def update(rows):
            return tracker.update(
                Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (200, 200))
            )

        first = update([[10, 10, 60, 100, 0.9, 0]])
        ident = int(first[0, 4])
        for _ in range(3):
            result = update([[10, 10, 60, 100, 0.15, 0], [130, 10, 180, 100, 0.15, 0]])
            self.assertEqual(result.shape[0], 1)
            self.assertEqual(int(result[0, 4]), ident)
        for _ in range(3):
            self.assertEqual(len(update([])), 0)
        restored = update([[10, 10, 60, 100, 0.9, 0]])
        self.assertEqual(int(restored[0, 4]), ident)


class PipelineAttributionTest(unittest.TestCase):
    def make_processor(self):
        from detection import MonitoringProcessor
        from identity import DetectedFace
        from pose import PoseResult

        self.pose = Mock()
        self.pose.analyze.return_value = [PoseResult(7, (0, 0, 80, 100), {})]
        self.matcher = Mock()
        face = DetectedFace(np.zeros(15), (10, 10, 30, 30), 0.95)
        self.matcher.detect_faces.return_value = [face]
        self.matcher.detect_largest_face.return_value = face
        self.matcher.is_face_usable.return_value = True
        self.matcher.identify.return_value = IdentityMatch("Alex", 0.8, True)
        with patch("detection.configure_logging"):
            processor = MonitoringProcessor(
                replace(AppConfig(), expression_enabled=False, identity_interval_frames=1),
                pose_analyzer=self.pose,
                identity_matcher=self.matcher,
                known_identities=[KnownIdentity("Alex", np.zeros(4), Path("fixture.jpg"))],
                event_recorder=EventRecorder(None),
            )
        self.addCleanup(processor.close)
        self.frame = np.zeros((200, 200, 3), np.uint8)
        for t in range(3):
            processor.process_frame(self.frame, timestamp=t)
        self.assertTrue(processor.last_snapshots[0].identity_confirmed)
        return processor

    def test_reentry_requires_fresh_identity_consensus(self):
        from pose import PoseResult

        processor = self.make_processor()
        self.pose.analyze.return_value = []
        processor.process_frame(self.frame, timestamp=3)
        self.pose.analyze.return_value = [PoseResult(7, (0, 0, 80, 100), {})]
        processor.process_frame(self.frame, timestamp=4)
        self.assertFalse(processor.last_snapshots[0].identity_confirmed)

    def test_crossing_body_boxes_suppress_identity_for_both_people(self):
        from pose import PoseResult

        processor = self.make_processor()
        self.pose.analyze.return_value = [
            PoseResult(7, (0, 0, 80, 100), {}),
            PoseResult(8, (20, 0, 100, 100), {}),
        ]
        processor.process_frame(self.frame, timestamp=3)
        self.assertTrue(all(not s.identity_confirmed for s in processor.last_snapshots))

    def test_multiple_faces_in_one_body_crop_clear_confirmation(self):
        processor = self.make_processor()
        self.matcher.detect_faces.return_value *= 2
        processor.process_frame(self.frame, timestamp=3)
        self.assertFalse(processor.last_snapshots[0].identity_confirmed)

    def test_cached_face_ownership_is_checked_between_recognition_frames(self):
        from detection import _resolve_identity_overlay

        processor = self.make_processor()
        runtime = processor.person_states[7]
        runtime.cached_face_box = (85, 10, 95, 30)
        runtime.last_face_seen_frame = 2
        overlay = _resolve_identity_overlay(
            frame=self.frame,
            person_box=(0, 0, 100, 200),
            identity_matcher=self.matcher,
            known_identities=processor.known_identities,
            runtime=runtime,
            frame_number=3,
            identity_interval_frames=5,
            other_person_boxes=[(80, 0, 180, 200)],
        )
        self.assertFalse(overlay.confirmed)

    def test_teleported_tracker_id_does_not_carry_name(self):
        from pose import PoseResult

        processor = self.make_processor()
        self.pose.analyze.return_value = [PoseResult(7, (110, 0, 190, 100), {})]
        processor.process_frame(self.frame, timestamp=3)
        self.assertFalse(processor.last_snapshots[0].identity_confirmed)
