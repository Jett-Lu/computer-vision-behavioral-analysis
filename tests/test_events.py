from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events import EventRecorder, TrackSnapshot


def snapshot(tier: str = "CLEAR", waves: int = 0) -> TrackSnapshot:
    return TrackSnapshot(
        track_key=7,
        track_id=7,
        identity_name="Person",
        identity_score=None,
        identity_confirmed=False,
        tier_label=tier,
        review_score=float(waves),
        wave_count=waves,
        expression_label=None,
        expression_context_strength=0.0,
    )


class EventRecorderTest(unittest.TestCase):
    def test_records_track_start_wave_and_tier_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(snapshot(), 0.0, 0)
            recorder.record(replace(snapshot("MONITOR", 3), wave_detected=True), 1.0, 1)
            recorder.close()

            events = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(
            [event["event_type"] for event in events],
            ["track_started", "tier_changed", "wave_counted", "track_ended"],
        )
        self.assertNotIn("activity_label", events[0])
        self.assertNotIn("activity_confidence", events[0])

    def test_wave_is_recorded_when_an_old_wave_expires_at_the_same_time(self) -> None:
        recorder = EventRecorder(None)
        recorder.record(snapshot(waves=1), 0.0, 0)
        recorder.record(replace(snapshot(waves=1), wave_detected=True), 30.1, 301)
        self.assertEqual(recorder.recent_messages[-1], "Track 7: wave 1")

    def test_demo_count_increase_does_not_fabricate_a_wave_event(self) -> None:
        recorder = EventRecorder(None)
        recorder.record(snapshot(), 0.0, 0)
        recorder.record(replace(snapshot("HIGH", 5), demo_override=True), 1.0, 1)
        self.assertFalse(any(": wave " in message for message in recorder.recent_messages))

    def test_expression_event_reports_context_strength(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(snapshot(), 0.0, 0)
            recorder.record(
                replace(
                    snapshot(),
                    expression_label="Anger",
                    expression_context_strength=0.30,
                ),
                1.0,
                1,
            )
            recorder.close()

            event = next(
                event
                for line in path.read_text().splitlines()
                if (event := json.loads(line))["event_type"] == "expression_changed"
            )

        self.assertEqual(event["event_type"], "expression_changed")
        self.assertEqual(event["details"], {"label": "Anger", "strength": 0.3})

    def test_activity_events_are_emitted_only_when_the_label_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(snapshot(), 0.0, 0)
            recorder.record(
                replace(
                    snapshot(),
                    activity_label="walking",
                    activity_confidence=0.82,
                ),
                1.0,
                1,
            )
            recorder.record(
                replace(
                    snapshot(),
                    activity_label="walking",
                    activity_confidence=0.91,
                ),
                2.0,
                2,
            )
            recorder.record(
                replace(
                    snapshot(),
                    activity_label="running",
                    activity_confidence=0.76,
                ),
                3.0,
                3,
            )
            recorder.close()

            events = [json.loads(line) for line in path.read_text().splitlines()]

        activity_events = [event for event in events if event["event_type"] == "activity_changed"]
        self.assertEqual(len(activity_events), 2)
        self.assertEqual(activity_events[0]["details"]["from"], None)
        self.assertEqual(activity_events[0]["details"]["to"], "walking")
        self.assertEqual(activity_events[1]["details"]["from"], "walking")
        self.assertEqual(activity_events[1]["details"]["to"], "running")


if __name__ == "__main__":
    unittest.main()
