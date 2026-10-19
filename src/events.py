"""Structured operator events for live and offline monitoring sessions."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logging_setup import get_logger
from data_policy import open_private_append

logger = get_logger("events")


@dataclass(frozen=True)
class TrackSnapshot:
    track_key: int
    track_id: int | None
    identity_name: str
    identity_score: float | None
    identity_confirmed: bool
    tier_label: str
    review_score: float
    wave_count: int
    expression_label: str | None
    expression_context_strength: float
    demo_override: bool = False
    activity_label: str | None = None
    activity_confidence: float | None = None
    wave_detected: bool = False


@dataclass(frozen=True)
class MonitorEvent:
    event_type: str
    timestamp_seconds: float
    frame_number: int
    track_key: int
    track_id: int | None
    identity_name: str
    details: dict[str, Any]
    activity_label: str | None = None
    activity_confidence: float | None = None


class EventRecorder:
    """Write meaningful track changes to JSONL and retain recent UI messages."""

    def __init__(
        self,
        output_path: Path | None,
        recent_limit: int = 5,
        *,
        identity_enabled: bool = True,
        expression_enabled: bool = True,
    ) -> None:
        self.identity_enabled = identity_enabled
        self.expression_enabled = expression_enabled
        self.output_path = output_path
        self._stream = None
        self._previous: dict[int, TrackSnapshot] = {}
        self._recent_messages: deque[str] = deque(maxlen=max(1, recent_limit))
        self._context: dict[str, Any] = {}
        if output_path is not None:
            self._stream = open_private_append(output_path)
            logger.info("Event report: %s", output_path)

    @property
    def recent_messages(self) -> tuple[str, ...]:
        return tuple(self._recent_messages)

    def record(
        self,
        snapshot: TrackSnapshot,
        timestamp_seconds: float,
        frame_number: int,
    ) -> None:
        previous = self._previous.get(snapshot.track_key)
        if previous is None:
            self._emit(
                "track_started",
                snapshot,
                timestamp_seconds,
                frame_number,
                {"tier": snapshot.tier_label},
                f"Track {snapshot.track_key} started",
            )
            if snapshot.demo_override:
                self._emit(
                    "demo_override_applied",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {},
                    f"Track {snapshot.track_key}: DEMO override",
                )
            if snapshot.activity_label is not None:
                self._emit_activity_change(
                    snapshot,
                    None,
                    timestamp_seconds,
                    frame_number,
                )
        else:
            if snapshot.tier_label != previous.tier_label:
                self._emit(
                    "tier_changed",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {
                        "from": previous.tier_label,
                        "to": snapshot.tier_label,
                        "score": round(snapshot.review_score, 4),
                    },
                    f"Track {snapshot.track_key}: {previous.tier_label} -> {snapshot.tier_label}",
                )
            if snapshot.wave_detected:
                self._emit(
                    "wave_counted",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {"wave_count": snapshot.wave_count},
                    f"Track {snapshot.track_key}: wave {snapshot.wave_count}",
                )
            if (
                self.identity_enabled
                and snapshot.identity_confirmed
                and snapshot.identity_name != previous.identity_name
            ):
                self._emit(
                    "identity_confirmed",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {"score": snapshot.identity_score},
                    f"Track {snapshot.track_key}: identity {snapshot.identity_name}",
                )
            if (
                self.expression_enabled
                and snapshot.expression_label is not None
                and snapshot.expression_label != previous.expression_label
            ):
                self._emit(
                    "expression_changed",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {
                        "label": snapshot.expression_label,
                        "strength": snapshot.expression_context_strength,
                    },
                    f"Track {snapshot.track_key}: expression {snapshot.expression_label}",
                )
            if snapshot.demo_override and not previous.demo_override:
                self._emit(
                    "demo_override_applied",
                    snapshot,
                    timestamp_seconds,
                    frame_number,
                    {},
                    f"Track {snapshot.track_key}: DEMO override",
                )
            if (
                snapshot.activity_label is not None
                and snapshot.activity_label != previous.activity_label
            ):
                self._emit_activity_change(
                    snapshot,
                    previous.activity_label,
                    timestamp_seconds,
                    frame_number,
                )
        self._previous[snapshot.track_key] = snapshot

    def end_track(
        self,
        track_key: int,
        timestamp_seconds: float,
        frame_number: int,
    ) -> None:
        previous = self._previous.pop(track_key, None)
        if previous is None:
            return
        self._emit(
            "track_ended",
            previous,
            timestamp_seconds,
            frame_number,
            {},
            f"Track {track_key} ended",
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def reset_tracks(self, **context: Any) -> None:
        """Start a new source within the same report without reloading models."""
        self._previous.clear()
        self._recent_messages.clear()
        self._context = dict(context)

    def _emit(
        self,
        event_type: str,
        snapshot: TrackSnapshot,
        timestamp_seconds: float,
        frame_number: int,
        details: dict[str, Any],
        message: str,
    ) -> None:
        event = MonitorEvent(
            event_type=event_type,
            timestamp_seconds=timestamp_seconds,
            frame_number=frame_number,
            track_key=snapshot.track_key,
            track_id=snapshot.track_id,
            identity_name=snapshot.identity_name,
            details=details,
            activity_label=snapshot.activity_label,
            activity_confidence=snapshot.activity_confidence,
        )
        self._recent_messages.append(message)
        if self._stream is not None:
            payload = asdict(event)
            if not self.identity_enabled:
                payload.pop("identity_name")
            if event.activity_label is None:
                payload.pop("activity_label")
                payload.pop("activity_confidence")
            payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
            payload["context"] = self._context
            self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self._stream.flush()

    def _emit_activity_change(
        self,
        snapshot: TrackSnapshot,
        previous_label: str | None,
        timestamp_seconds: float,
        frame_number: int,
    ) -> None:
        self._emit(
            "activity_changed",
            snapshot,
            timestamp_seconds,
            frame_number,
            {
                "from": previous_label,
                "to": snapshot.activity_label,
                "confidence": snapshot.activity_confidence,
            },
            f"Track {snapshot.track_key}: activity {snapshot.activity_label}",
        )


def session_event_path(event_dir: Path, prefix: str = "live") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return event_dir / f"{prefix}_{timestamp}.jsonl"
