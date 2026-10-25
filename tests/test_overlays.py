from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import AppConfig, OverlayConfig
from events import TrackSnapshot
from pose import PoseResult
from overlays import OverlayRenderer, RenderPerson, person_lines
from detection import MonitoringProcessor
from events import EventRecorder
from unittest.mock import Mock


def snapshot(track_id=7, **changes):
    return replace(
        TrackSnapshot(7, track_id, "Person", None, False, "CLEAR", 0, 0, None, 0), **changes
    )


def person(track_id=7, box=(40, 100, 180, 300), **changes):
    return RenderPerson(
        PoseResult(track_id, box, {5: (0.2, 0.4), 6: (0.3, 0.4), 10: (0.35, 0.3)}),
        snapshot(track_id, **changes),
        ((0.2, 0.6), (0.25, 0.6)),
    )


def hidden():
    return OverlayConfig(
        boxes=False,
        skeletons=False,
        track_ids=False,
        labels=False,
        activity=False,
        trails=False,
        hud=False,
    )


class OverlayTest(unittest.TestCase):
    def render(self, settings=None, people=None, shape=(360, 640, 3), **kwargs):
        frame = np.full(shape, 30, np.uint8)
        renderer = OverlayRenderer(
            settings or OverlayConfig(), identity_enabled=False, expression_enabled=False
        )
        panels = renderer.draw(
            frame, people if people is not None else [person()], fps=30, recent_events=(), **kwargs
        )
        return frame, panels

    def test_settings_parse_independently_and_bound_resources(self):
        with patch.dict(
            os.environ,
            {
                "MONITOR_OVERLAY_BOXES": "false",
                "MONITOR_OVERLAY_TRAILS": "true",
                "MONITOR_OVERLAY_TRAIL_POINTS": "99999",
                "MONITOR_OVERLAY_PANEL_ALPHA": "-2",
                "MONITOR_OVERLAY_TEXT_SCALE": "nan",
            },
        ):
            cfg = AppConfig.from_env()
        self.assertFalse(cfg.overlays.boxes)
        self.assertTrue(cfg.overlays.trails)
        self.assertEqual(cfg.overlays.trail_points, 300)
        self.assertEqual(cfg.overlays.panel_alpha, 0)
        self.assertEqual(cfg.overlays.text_scale, 1)

    def test_track_ids_are_not_identity_and_pending_has_no_fake_id(self):
        lines = person_lines(snapshot(), OverlayConfig(), False, False)
        self.assertIn("Track 7", lines)
        self.assertTrue(any("DEMO" in line for line in lines))
        self.assertIn("Track pending", person_lines(snapshot(None), OverlayConfig(), False, False))

    def test_confirmed_identity_preserves_match_score(self):
        state = snapshot(identity_name="Alex", identity_confirmed=True, identity_score=0.83)
        self.assertIn("Identity: Alex 0.83", person_lines(state, OverlayConfig(), True, False))
        self.assertNotIn("Alex", str(person_lines(state, OverlayConfig(), False, False)))

    def test_privacy_redacts_person_labels_even_with_sensitive_input(self):
        state = snapshot(
            identity_name="Private Name",
            identity_confirmed=True,
            display_expression_label="Anger",
            display_expression_confidence=0.9,
        )
        lines = person_lines(state, OverlayConfig(), False, False)
        self.assertNotIn("Private Name", str(lines))
        self.assertNotIn("Anger", str(lines))

    def test_all_layers_off_is_pixel_identical_and_each_layer_is_independent(self):
        frame, panels = self.render(hidden())
        self.assertTrue(np.all(frame == 30))
        self.assertEqual(panels, [])
        for name in ("boxes", "skeletons", "track_ids", "labels", "activity", "trails", "hud"):
            with self.subTest(layer=name):
                frame, _ = self.render(replace(hidden(), **{name: True}))
                self.assertTrue(np.any(frame != 30))

    def test_layout_stays_in_frame_and_avoids_overlapping_panels(self):
        for shape in ((1, 1, 3), (24, 40, 3), (120, 160, 3), (720, 1280, 3)):
            frame, panels = self.render(
                people=[person(i, (0, 0, 60, 90)) for i in range(8)], shape=shape
            )
            self.assertEqual(frame.shape, shape)
            for i, panel in enumerate(panels):
                x1, y1, x2, y2 = panel.box
                self.assertTrue(0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0])
                for other in panels[:i]:
                    a, b, c, d = other.box
                    self.assertFalse(x1 < c and x2 > a and y1 < d and y2 > b)

    def test_panel_prefers_body_top_when_hud_blocks_space_above(self):
        _, panels = self.render(people=[person(7, (40, 60, 180, 350))])
        panel = next(p for p in panels if p.track_id == 7)
        self.assertLess(panel.box[1], 200)

    def test_small_frame_keeps_a_compact_hud_row(self):
        _, panels = self.render(shape=(120, 160, 3))
        self.assertTrue(any(line.startswith("FPS") for p in panels for line in p.lines))

    def test_known_label_is_never_truncated_into_a_different_name(self):
        _, panels = self.render(
            people=[
                person(identity_name="Very long identity name with suffix", identity_confirmed=True)
            ],
            shape=(200, 200, 3),
        )
        # Anonymous renderer must omit it entirely, including truncated variants.
        self.assertFalse(any("Identity:" in line for p in panels for line in p.lines))

    def test_deterministic_render_missing_keypoints_and_input_state_unchanged(self):
        p = person()
        p.pose.landmarks.clear()
        first, panels = self.render(people=[p])
        second, again = self.render(people=[p])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(panels, again)
        self.assertEqual(p.snapshot, snapshot())

    def test_unknown_and_unavailable_are_distinct(self):
        unknown = person_lines(snapshot(identity_name="Unknown"), OverlayConfig(), True, True)
        missing = person_lines(snapshot(), OverlayConfig(), True, True)
        self.assertIn("Identity: unknown", unknown)
        self.assertIn("Identity: unavailable", missing)
        self.assertIn("Activity: unavailable", missing)
        self.assertIn("Expression: unavailable", missing)

    def test_private_recent_messages_are_not_rendered_in_anonymous_mode(self):
        renderer = OverlayRenderer(
            OverlayConfig(), identity_enabled=False, expression_enabled=False
        )
        panels = renderer.draw(
            np.zeros((360, 640, 3), np.uint8),
            [],
            fps=30,
            recent_events=(
                "Track 7: identity Private Name",
                "Track 7: expression Anger",
                "Track 7: identity cleared",
            ),
        )
        self.assertNotIn("Private Name", str(panels))
        self.assertNotIn("Anger", str(panels))


class OverlayPipelineTest(unittest.TestCase):
    def test_synthetic_photo_and_video_export_are_annotated_and_readable(self):
        import cv2

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from overlay_samples import create_samples

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("detection.configure_logging"):
                result = create_samples(root, frame_count=6)
            photo = cv2.imread(str(root / "sample_annotated.png"))
            raw = cv2.imread(str(root / "sample.png"))
            self.assertEqual(photo.shape, raw.shape)
            self.assertFalse(np.array_equal(photo, raw))
            capture = cv2.VideoCapture(str(root / "sample_annotated.mp4"))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
                ok, video = capture.read()
                self.assertTrue(ok)
                self.assertEqual(video.shape, raw.shape)
            finally:
                capture.release()
            self.assertEqual(result["frames"], 6)

    def processor(self, settings=None, path=None):
        pose = Mock()
        pose.analyze.return_value = [person().pose]
        cfg = replace(
            AppConfig(),
            identity_enabled=False,
            expression_enabled=False,
            overlays=settings or OverlayConfig(trails=True, trail_points=3),
            stale_track_frames=2,
        )
        with patch("detection.configure_logging"):
            processor = MonitoringProcessor(
                cfg, pose_analyzer=pose, event_recorder=EventRecorder(path)
            )
        self.addCleanup(processor.close)
        return processor, pose

    def test_display_controls_do_not_change_snapshots_or_event_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            payloads = []
            for index, settings in enumerate((OverlayConfig(trails=True), hidden())):
                path = Path(directory) / f"{index}.jsonl"
                processor, pose = self.processor(settings, path)
                original = np.full((360, 640, 3), 30, np.uint8)
                for i in range(4):
                    output = processor.process_frame(original, timestamp=i * 0.1)
                self.assertTrue(np.all(original == 30))
                if index == 1:
                    np.testing.assert_array_equal(output, original)
                processor.close()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    row.pop("recorded_at")
                    row.pop("session_id")
                payloads.append((processor.last_snapshots, rows, pose.analyze.call_count))
            self.assertEqual(payloads[0], payloads[1])

    def test_trails_are_bounded_and_clear_on_discontinuity_expiry_and_reset(self):
        processor, pose = self.processor()
        frame = np.zeros((360, 640, 3), np.uint8)
        for i in range(6):
            pose.analyze.return_value = [PoseResult(7, (40 + i, 100, 180 + i, 300), {})]
            processor.process_frame(frame, timestamp=i * 0.1)
        self.assertEqual(len(processor.person_states[7].trail), 3)
        pose.analyze.return_value = []
        processor.process_frame(frame, timestamp=0.6)
        pose.analyze.return_value = [person().pose]
        processor.process_frame(frame, timestamp=0.7)
        self.assertEqual(len(processor.person_states[7].trail), 1)
        pose.analyze.return_value = [PoseResult(7, (400, 100, 500, 300), {})]
        processor.process_frame(frame, timestamp=0.8)
        self.assertEqual(len(processor.person_states[7].trail), 1)
        pose.analyze.return_value = []
        for i in range(4):
            processor.process_frame(frame, timestamp=1 + i * 0.1)
        self.assertNotIn(7, processor.person_states)
        processor.reset_tracking(source="new synthetic source")
        self.assertFalse(processor.person_states)

    def test_two_tracks_have_independent_trails_and_overlap_breaks_them(self):
        processor, pose = self.processor()
        frame = np.zeros((360, 640, 3), np.uint8)
        pose.analyze.return_value = [person().pose, PoseResult(8, (400, 100, 500, 300), {})]
        for i in range(3):
            processor.process_frame(frame, timestamp=i * 0.1)
        self.assertNotEqual(processor.person_states[7].trail, processor.person_states[8].trail)
        pose.analyze.return_value = [person().pose, PoseResult(8, (50, 100, 190, 300), {})]
        processor.process_frame(frame, timestamp=0.3)
        self.assertTrue(all(len(s.trail) <= 1 for s in processor.person_states.values()))


if __name__ == "__main__":
    unittest.main()
