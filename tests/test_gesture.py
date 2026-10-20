from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture import RightHandWaveMonitor
from pose import MoveNetKeypoint


def raised_right_hand(x: float) -> dict[int, tuple[float, float]]:
    return {
        MoveNetKeypoint.RIGHT_SHOULDER: (0.50, 0.50),
        MoveNetKeypoint.LEFT_SHOULDER: (0.30, 0.50),
        MoveNetKeypoint.RIGHT_WRIST: (x, 0.35),
    }


def low_right_hand(x: float) -> dict[int, tuple[float, float]]:
    return {
        MoveNetKeypoint.RIGHT_SHOULDER: (0.50, 0.50),
        MoveNetKeypoint.LEFT_SHOULDER: (0.30, 0.50),
        MoveNetKeypoint.RIGHT_WRIST: (x, 0.70),
    }


def complete_wave(monitor: RightHandWaveMonitor, start_time: float) -> None:
    for offset, x in enumerate((0.42, 0.62, 0.38, 0.64)):
        monitor.update(raised_right_hand(x), start_time + offset * 0.15)


class RightHandWaveMonitorTest(unittest.TestCase):
    def test_body_translation_does_not_count_as_hand_motion(self) -> None:
        monitor = RightHandWaveMonitor()
        for index, x in enumerate((0.3, 0.4, 0.3, 0.4)):
            state = monitor.update(
                {
                    MoveNetKeypoint.RIGHT_SHOULDER: (x, 0.5),
                    MoveNetKeypoint.LEFT_SHOULDER: (x - 0.2, 0.5),
                    MoveNetKeypoint.RIGHT_WRIST: (x + 0.1, 0.3),
                },
                index * 0.3,
            )
        self.assertFalse(state.wave_detected)
        self.assertEqual(state.recent_wave_count, 0)

    def test_low_hand_motion_does_not_count_as_wave(self) -> None:
        monitor = RightHandWaveMonitor()
        for offset, x in enumerate((0.42, 0.62, 0.38, 0.64)):
            state = monitor.update(low_right_hand(x), offset * 0.15)

        self.assertEqual(state.recent_wave_count, 0)

    def test_one_or_two_waves_remain_clear_and_green(self) -> None:
        monitor = RightHandWaveMonitor()
        complete_wave(monitor, 0.0)
        complete_wave(monitor, 1.2)

        state = monitor.update({}, 2.0)

        self.assertEqual(state.recent_wave_count, 2)

    def test_repeated_waves_raise_tier_and_shift_color_to_red(self) -> None:
        monitor = RightHandWaveMonitor()
        for wave_number in range(7):
            complete_wave(monitor, wave_number * 1.2)

        state = monitor.update({}, 9.0)

        self.assertEqual(state.recent_wave_count, 7)

    def test_wave_events_expire_and_indicator_recovers(self) -> None:
        monitor = RightHandWaveMonitor(event_window_seconds=5.0)
        for wave_number in range(4):
            complete_wave(monitor, wave_number * 1.0)

        state = monitor.update({}, 10.0)

        self.assertEqual(state.recent_wave_count, 0)


if __name__ == "__main__":
    unittest.main()
