from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from pose import MoveNetKeypoint


@dataclass(frozen=True)
class WaveAlertState:
    recent_wave_count: int
    wave_detected: bool


@dataclass
class RightHandWaveMonitor:
    """Track repeated raised right-hand waves in a rolling time window."""

    event_window_seconds: float = 30.0
    motion_window_seconds: float = 2.5
    event_cooldown_seconds: float = 0.8
    horizontal_movement_threshold: float = 0.25
    raised_hand_margin: float = 0.15
    max_observation_gap_seconds: float = 0.5
    _last_timestamp: float | None = field(default=None, init=False)
    _events: deque[float] = field(default_factory=deque, init=False)
    _anchor_x: float | None = field(default=None, init=False)
    _last_direction: int | None = field(default=None, init=False)
    _direction_changes: int = field(default=0, init=False)
    _motion_start_time: float | None = field(default=None, init=False)

    def update(
        self,
        landmarks: dict[int, tuple[float, float]],
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> WaveAlertState:
        self._discard_expired_events(timestamp)
        if not math.isfinite(timestamp):
            self._reset_motion()
            return self._state(wave_detected=False)
        if self._last_timestamp is not None and (
            timestamp <= self._last_timestamp
            or timestamp - self._last_timestamp > self.max_observation_gap_seconds
        ):
            self._reset_motion()
        self._last_timestamp = timestamp

        right_wrist = landmarks.get(MoveNetKeypoint.RIGHT_WRIST)
        right_shoulder = landmarks.get(MoveNetKeypoint.RIGHT_SHOULDER)
        left_shoulder = landmarks.get(MoveNetKeypoint.LEFT_SHOULDER)
        if right_wrist is None or right_shoulder is None or left_shoulder is None:
            self._reset_motion()
            return self._state(wave_detected=False)

        wrist_x, wrist_y = right_wrist
        shoulder_x, shoulder_y = right_shoulder
        aspect = frame_shape[1] / frame_shape[0] if frame_shape is not None else 1.0
        scale = math.hypot((left_shoulder[0] - shoulder_x) * aspect, left_shoulder[1] - shoulder_y)
        if (
            not all(math.isfinite(v) for v in (*right_wrist, *right_shoulder, *left_shoulder))
            or scale < 1e-4
        ):
            self._reset_motion()
            return self._state(wave_detected=False)
        wrist_x = (wrist_x - shoulder_x) * aspect / scale
        if (wrist_y - shoulder_y) / scale > self.raised_hand_margin:
            self._reset_motion()
            return self._state(wave_detected=False)

        if self._motion_start_time is None or self._anchor_x is None:
            self._start_motion(wrist_x, timestamp)
            return self._state(wave_detected=False)

        if timestamp - self._motion_start_time > self.motion_window_seconds:
            self._start_motion(wrist_x, timestamp)
            return self._state(wave_detected=False)

        delta_x = wrist_x - self._anchor_x
        if abs(delta_x) < self.horizontal_movement_threshold:
            return self._state(wave_detected=False)

        direction = 1 if delta_x > 0 else -1
        if self._last_direction is not None and direction != self._last_direction:
            self._direction_changes += 1

        self._last_direction = direction
        self._anchor_x = wrist_x

        cooled_down = not self._events or (
            timestamp - self._events[-1] >= self.event_cooldown_seconds
        )
        if self._direction_changes >= 2 and cooled_down:
            self._events.append(timestamp)
            self._start_motion(wrist_x, timestamp)
            return self._state(wave_detected=True)

        return self._state(wave_detected=False)

    def _start_motion(self, wrist_x: float, timestamp: float) -> None:
        self._anchor_x = wrist_x
        self._last_direction = None
        self._direction_changes = 0
        self._motion_start_time = timestamp

    def _reset_motion(self) -> None:
        self._anchor_x = None
        self._last_direction = None
        self._direction_changes = 0
        self._motion_start_time = None

    def _discard_expired_events(self, timestamp: float) -> None:
        while self._events and timestamp - self._events[0] > self.event_window_seconds:
            self._events.popleft()

    def _state(self, wave_detected: bool) -> WaveAlertState:
        return WaveAlertState(
            recent_wave_count=len(self._events),
            wave_detected=wave_detected,
        )
