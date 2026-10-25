"""OpenCV presentation only: no model calls, identity decisions or event writes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from config import OverlayConfig
from events import TrackSnapshot
from pose import POSE_CONNECTIONS, PoseResult

Box = tuple[int, int, int, int]
Color = tuple[int, int, int]
PALETTE: tuple[Color, ...] = (
    (235, 195, 70),
    (100, 215, 245),
    (210, 140, 235),
    (145, 225, 130),
    (245, 175, 120),
    (160, 165, 245),
)


@dataclass(frozen=True)
class RenderPerson:
    pose: PoseResult
    snapshot: TrackSnapshot
    trail: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Panel:
    box: Box
    lines: tuple[str, ...]
    track_id: int | None = None


def person_lines(
    state: TrackSnapshot, settings: OverlayConfig, identity_enabled: bool, expression_enabled: bool
) -> list[str]:
    lines: list[str] = []
    if settings.track_ids:
        lines.append(f"Track {state.track_id}" if state.track_id is not None else "Track pending")
    if settings.activity:
        lines.append(
            f"Activity: {state.activity_label.title()} {state.activity_confidence:.0%}"
            if state.activity_label is not None and state.activity_confidence is not None
            else "Activity: unavailable"
        )
    if settings.labels:
        lines.append(f"DEMO {state.tier_label} | waves {state.wave_count}")
        if state.demo_override and identity_enabled:
            lines.append("DEMO OVERRIDE")
        if identity_enabled:
            if state.identity_confirmed:
                score = f" {state.identity_score:.2f}" if state.identity_score is not None else ""
                lines.append(f"Identity: {state.identity_name}{score}")
            else:
                lines.append(
                    "Identity: unknown"
                    if state.identity_name == "Unknown"
                    else "Identity: unavailable"
                )
        if expression_enabled:
            lines.append(
                f"Expression: {state.display_expression_label} {state.display_expression_confidence:.0%}"
                if state.display_expression_label is not None
                and state.display_expression_confidence is not None
                else "Expression: unavailable"
            )
    return lines


class OverlayRenderer:
    def __init__(
        self, settings: OverlayConfig, *, identity_enabled: bool, expression_enabled: bool
    ) -> None:
        self.settings = settings
        self.identity_enabled = identity_enabled
        self.expression_enabled = expression_enabled

    def draw(
        self,
        frame: np.ndarray,
        people: Sequence[RenderPerson],
        *,
        fps: float,
        recent_events: Sequence[str],
    ) -> list[Panel]:
        """Draw in place and return fitted panel geometry for deterministic testing."""
        height, width = frame.shape[:2]
        if height < 2 or width < 2:
            return []
        options = self.settings
        thickness = max(1, min(8, options.line_thickness))
        scale = max(0.35, min(0.8, min(width / 1280, height / 720) * 0.6)) * max(
            0.5, min(2, options.text_scale)
        )
        panels: list[Panel] = []
        # Geometry first, panels last: skeletons and trails never overwrite text.
        for person in people:
            color = (
                PALETTE[person.snapshot.track_id % len(PALETTE)]
                if person.snapshot.track_id is not None
                else (175, 175, 175)
            )
            if options.trails and person.snapshot.track_id is not None:
                trail = [
                    self._point(p, width, height)
                    for p in person.trail
                    if all(math.isfinite(v) for v in p)
                ]
                for i in range(1, len(trail)):
                    fade = 0.3 + 0.7 * i / max(1, len(trail) - 1)
                    cv2.line(
                        frame,
                        trail[i - 1],
                        trail[i],
                        tuple(int(c * fade) for c in color),
                        thickness,
                        cv2.LINE_AA,
                    )
            if options.boxes:
                x1, y1, x2, y2 = person.pose.box
                cv2.rectangle(
                    frame,
                    (max(0, x1), max(0, y1)),
                    (min(width - 1, x2), min(height - 1, y2)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            if options.skeletons:
                points = {
                    key: self._point(point, width, height)
                    for key, point in person.pose.landmarks.items()
                    if all(math.isfinite(v) for v in point)
                }
                for start, end in POSE_CONNECTIONS:
                    if start in points and end in points:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (20, 25, 30),
                            thickness + 2,
                            cv2.LINE_AA,
                        )
                        cv2.line(frame, points[start], points[end], color, thickness, cv2.LINE_AA)
                for point in points.values():
                    cv2.circle(frame, point, thickness + 1, (245, 245, 245), -1, cv2.LINE_AA)
                    cv2.circle(frame, point, thickness + 1, color, 1, cv2.LINE_AA)
        if options.hud:
            tracked = sum(p.snapshot.track_id is not None for p in people)
            hud = [
                f"FPS {fps:.1f} | tracked {tracked} | pending {len(people) - tracked}",
                "DEMO review | IDs are temporary tracks",
            ]
            safe_events = [
                event
                for event in recent_events
                if (self.identity_enabled or "identity" not in event.casefold())
                and (self.expression_enabled or "expression" not in event.casefold())
            ]
            hud.extend(safe_events[-2:])
            self._panel(
                frame, hud, (0, 0, width, 0), (160, 170, 180), panels, scale, None, hud=True
            )
        for person in people:
            color = (
                PALETTE[person.snapshot.track_id % len(PALETTE)]
                if person.snapshot.track_id is not None
                else (175, 175, 175)
            )
            lines = person_lines(
                person.snapshot, options, self.identity_enabled, self.expression_enabled
            )
            self._panel(
                frame, lines, person.pose.box, color, panels, scale, person.snapshot.track_id
            )
        return panels

    @staticmethod
    def _point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
        return (
            int(max(0, min(width - 1, point[0] * width))),
            int(max(0, min(height - 1, point[1] * height))),
        )

    def _panel(
        self,
        frame: np.ndarray,
        lines: list[str],
        anchor: Box,
        color: Color,
        panels: list[Panel],
        scale: float,
        track_id: int | None,
        *,
        hud: bool = False,
    ) -> None:
        if not lines:
            return
        height, width = frame.shape[:2]
        padding = 4
        text_height = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][1]
        line_height = text_height + 8
        # HUD occupies at most one quarter of the frame; tiny frames omit it.
        max_lines = min(len(lines), (height // 4 if hud else height) // line_height - 1)
        if hud and height >= 80 and width >= 96:
            max_lines = max(1, max_lines)
        if max_lines < 1 or width < 32:
            return
        max_width = min(width, max(160, width // 2))
        fitted = [self._fit(line, max_width - padding * 2, scale) for line in lines[:max_lines]]
        fitted = [line for line in fitted if line]
        while fitted:
            panel_width = min(
                width,
                max(
                    cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                    for line in fitted
                )
                + padding * 2,
            )
            panel_height = line_height * len(fitted) + padding
            x1, y1, x2, y2 = anchor
            preferred = (
                [(x1, y1 - panel_height - 3), (x1, y1)]
                + [(x1, max(y1, panel.box[3] + 3)) for panel in panels]
                + [(x1, y2 + 3)]
                if not hud
                else [(0, 0)]
            )
            preferred.extend(
                (x, y)
                for y in range(0, height - panel_height + 1, line_height)
                for x in (x1, x2 + 3, x1 - panel_width - 3, 0, width - panel_width)
            )
            for x, y in preferred:
                x = max(0, min(width - panel_width, x))
                y = max(0, min(height - panel_height, y))
                box = (x, y, x + panel_width, y + panel_height)
                if any(
                    box[0] < p.box[2]
                    and box[2] > p.box[0]
                    and box[1] < p.box[3]
                    and box[3] > p.box[1]
                    for p in panels
                ):
                    continue
                roi = frame[y : y + panel_height, x : x + panel_width]
                alpha = max(0, min(1, self.settings.panel_alpha))
                cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, (24, 29, 36)), alpha, 0, dst=roi)
                cv2.line(frame, (x, y), (x, y + panel_height - 1), color, 2)
                for index, line in enumerate(fitted):
                    cv2.putText(
                        frame,
                        line,
                        (x + padding, y + padding + text_height + index * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (245, 247, 250),
                        1,
                        cv2.LINE_AA,
                    )
                panels.append(Panel(box, tuple(fitted), track_id))
                return
            # Prefer fewer lines over overlapping labels on crowded frames.
            fitted.pop()

    @staticmethod
    def _fit(text: str, width: int, scale: float) -> str:
        text = "".join(c if 32 <= ord(c) <= 126 else "?" for c in text)
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width:
            return text
        while (
            text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width
        ):
            text = text[:-1]
        return text + "..." if text else ""
