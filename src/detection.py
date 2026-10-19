"""Live multi-person monitoring with optional enrolled-person identification."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import cv2

from camera import CaptureClock, open_capture
from config import DEFAULT_CONFIG, AppConfig
from events import EventRecorder, TrackSnapshot, session_event_path
from identity import IdentityConsensus, KnownIdentity, OpenCVFaceIdentifier, offset_box
from logging_setup import configure_logging, get_logger
from model_manager import (
    ModelUnavailableError,
    emotion_model_spec,
    ensure_model,
    ensure_models,
    identity_model_specs,
)
from review import HIGH_COLOR, ReviewState

TMP_DIR = DEFAULT_CONFIG.project_root / ".tmp"

logger = get_logger("detection")
_AUTO = object()
GLOBAL_HUD_WIDTH = 520
GLOBAL_HUD_MAX_HEIGHT = 60


@dataclass
class PersonRuntime:
    """State that must stay isolated for each tracked person."""

    wave_monitor: Any
    review_monitor: Any
    identity_consensus: IdentityConsensus
    emotion_smoother: Any = None
    cached_emotion: Any = None
    cached_face_box: tuple[int, int, int, int] | None = None
    last_face_seen_frame: int = -1
    last_seen_frame: int = 0


@dataclass(frozen=True)
class IdentityOverlay:
    """Resolved identity label details for one tracked person."""

    name: str
    face_box: tuple[int, int, int, int]
    label_text: str
    score: float | None = None
    confirmed: bool = False
    face_visible: bool = False


class StageTimer:
    """Accumulate and periodically report per-stage frame processing time."""

    def __init__(self, print_every: int = 30) -> None:
        self.print_every = print_every
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._frames = 0

    @contextmanager
    def __call__(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage_name] += time.perf_counter() - start
            self._counts[stage_name] += 1

    def tick(self) -> None:
        self._frames += 1
        if self._frames % self.print_every != 0:
            return

        rows = [
            (stage, (total / self._counts[stage]) * 1000, self._counts[stage])
            for stage, total in self._totals.items()
        ]
        rows.sort(key=lambda row: row[1], reverse=True)
        for stage, average_ms, calls in rows:
            logger.info("Timing %-18s %6.1f ms (%s calls)", stage, average_ms, calls)

        per_frame_total_ms = sum(total * 1000 / self.print_every for total in self._totals.values())
        if per_frame_total_ms > 0:
            logger.info(
                "Timing total %.1f ms/frame (~%.1f FPS upper bound)",
                per_frame_total_ms,
                1000 / per_frame_total_ms,
            )
        self._totals.clear()
        self._counts.clear()


class NoOpTimer:
    @contextmanager
    def __call__(self, stage_name: str):
        yield

    def tick(self) -> None:
        pass


def _track_key(
    track_id: int | None,
    detection_index: int,
    frame_number: int = 0,
) -> int:
    """Use persistent tracker IDs and frame-local keys when no ID exists yet."""
    if track_id is not None:
        return track_id
    return -((frame_number + 1) * 10_000 + detection_index + 1)


def _parse_demo_high_review_names(raw_names: str) -> set[str]:
    return {
        name.strip().casefold() for name in raw_names.replace(";", ",").split(",") if name.strip()
    }


def _apply_demo_review_override(
    review_state: ReviewState,
    identity_name: str,
    demo_high_review_names: set[str],
) -> ReviewState:
    if identity_name.casefold() not in demo_high_review_names:
        return review_state

    return replace(
        review_state,
        tier_label="HIGH",
        color=HIGH_COLOR,
        score=max(review_state.score, 5.0),
        recent_wave_count=max(review_state.recent_wave_count, 5),
    )


class MonitoringProcessor:
    """Apply the monitoring pipeline to frames from a live or recorded source."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        pose_analyzer: Any = None,
        emotion_analyzer: Any = _AUTO,
        identity_matcher: Any = _AUTO,
        activity_recognizer: Any = _AUTO,
        known_identities: Sequence[KnownIdentity] | None = None,
        event_recorder: EventRecorder | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        configure_logging(self.config.log_level, self.config.log_dir)

        self.identity_matcher = self._prepare_identity_matcher(identity_matcher)
        if not self.config.identity_enabled:
            self.known_identities = []
        elif known_identities is not None:
            self.known_identities = list(known_identities)
        elif self.identity_matcher is not None:
            self.known_identities = self.identity_matcher.load_enrollments(
                self.config.enrollments_dir
            )
        else:
            self.known_identities = []
        self._report_identity_mode()

        from gesture import RightHandWaveMonitor
        from pose import PoseAnalyzer
        from review import ReviewLevelMonitor

        logger.info("Loading multi-person pose model")
        self.pose_analyzer = (
            pose_analyzer
            if pose_analyzer is not None
            else PoseAnalyzer(
                model_path=self.config.pose_model_path,
                config=self.config,
            )
        )
        self.activity_recognizer = self._prepare_activity_recognizer(activity_recognizer)
        self.emotion_analyzer = self._prepare_emotion_analyzer(emotion_analyzer)
        if self.emotion_analyzer is not None:
            from emotion import EmotionSmoother

            self.emotion_smoother_type = EmotionSmoother
        else:
            self.emotion_smoother_type = None
        self.wave_monitor_type = RightHandWaveMonitor
        self.review_monitor_type = ReviewLevelMonitor
        self.person_states: dict[int, PersonRuntime] = {}
        self.demo_high_review_names = _parse_demo_high_review_names(
            self.config.demo_high_review_names
        )
        if self.demo_high_review_names:
            logger.warning(
                "DEMO override enabled for %s enrolled name(s)",
                len(self.demo_high_review_names),
            )

        if event_recorder is not None:
            self.event_recorder = event_recorder
            if isinstance(event_recorder, EventRecorder):
                event_recorder.identity_enabled = (
                    event_recorder.identity_enabled and self.config.identity_enabled
                )
                event_recorder.expression_enabled = (
                    event_recorder.expression_enabled and self.config.expression_enabled
                )
        else:
            event_path = (
                session_event_path(self.config.event_dir)
                if self.config.event_logging_enabled
                else None
            )
            self.event_recorder = EventRecorder(
                event_path,
                identity_enabled=self.config.identity_enabled,
                expression_enabled=self.config.expression_enabled,
            )

        self.frame_count = 0
        self._timestamp_origin: float | None = None
        self.last_snapshots: list[TrackSnapshot] = []
        self.timer = StageTimer(print_every=30) if self.config.debug_timing else NoOpTimer()
        self._fps = 0.0
        self._last_frame_clock: float | None = None

    @property
    def identity_enabled(self) -> bool:
        return self.identity_matcher is not None and bool(self.known_identities)

    @property
    def activity_enabled(self) -> bool:
        return self.activity_recognizer is not None

    def process_frame(
        self,
        frame: Any,
        timestamp: float | None = None,
        force_identity: bool = False,
    ) -> Any:
        if timestamp is None:
            timestamp = time.monotonic()
        if self._timestamp_origin is None:
            self._timestamp_origin = timestamp
        timestamp = max(0.0, timestamp - self._timestamp_origin)
        annotated = frame.copy()
        self._update_fps()

        with self.timer("pose_track"):
            tracked_poses = self.pose_analyzer.analyze(frame)

        snapshots: list[TrackSnapshot] = []
        active_keys: set[int] = set()
        demo_override_active = False
        for detection_index, pose_result in enumerate(tracked_poses):
            track_key = _track_key(
                pose_result.track_id,
                detection_index,
                self.frame_count,
            )
            active_keys.add(track_key)
            runtime = self.person_states.get(track_key)
            if runtime is None:
                runtime = self._new_person_runtime()
                self.person_states[track_key] = runtime
            runtime.last_seen_frame = self.frame_count

            activity_prediction = None
            if self.activity_recognizer is not None and pose_result.track_id is not None:
                with self.timer("activity"):
                    activity_prediction = self.activity_recognizer.update(
                        track_key,
                        pose_result.landmarks,
                        pose_result.box,
                        frame.shape,
                        self.frame_count,
                        timestamp=timestamp,
                    )

            wave_state = runtime.wave_monitor.update(
                pose_result.landmarks,
                timestamp,
            )

            context_activated = False
            if self.frame_count % self.config.expression_interval_frames == 0:
                detected_emotion = None
                if self.emotion_analyzer is not None:
                    with self.timer("emotion"):
                        detected_emotion = self.emotion_analyzer.analyze(
                            frame,
                            pose_result.box,
                        )
                runtime.cached_emotion = (
                    runtime.emotion_smoother.update(detected_emotion, timestamp)
                    if runtime.emotion_smoother is not None
                    else None
                )
                context_activated = runtime.review_monitor.observe_expression(
                    runtime.cached_emotion.label if runtime.cached_emotion is not None else None,
                    runtime.cached_emotion.confidence
                    if runtime.cached_emotion is not None
                    else None,
                )

            review_state = runtime.review_monitor.update(wave_state)
            with self.timer("identity"):
                identity_overlay = _resolve_identity_overlay(
                    frame=frame,
                    person_box=pose_result.box,
                    runtime=runtime,
                    frame_number=self.frame_count,
                    identity_interval_frames=(
                        1 if force_identity else self.config.identity_interval_frames
                    ),
                    identity_matcher=self.identity_matcher,
                    known_identities=self.known_identities,
                )

            demo_override = (
                identity_overlay.confirmed
                and identity_overlay.name.casefold() in self.demo_high_review_names
            )
            if demo_override:
                review_state = _apply_demo_review_override(
                    review_state,
                    identity_overlay.name,
                    self.demo_high_review_names,
                )
            demo_override_active = demo_override_active or demo_override

            with self.timer("render"):
                self.pose_analyzer.draw_landmarks(
                    annotated,
                    pose_result.landmarks,
                    review_state.color,
                )
                _draw_review_overlay(
                    annotated,
                    pose_result.box,
                    review_state,
                    wave_state.wave_detected,
                    context_activated,
                    (None if activity_prediction is None else activity_prediction.label),
                    (None if activity_prediction is None else activity_prediction.confidence),
                )
                _draw_identity_overlay(
                    annotated,
                    identity_overlay,
                    review_state.color,
                )

            snapshot = TrackSnapshot(
                track_key=track_key,
                track_id=pose_result.track_id,
                identity_name=identity_overlay.name,
                identity_score=identity_overlay.score,
                identity_confirmed=identity_overlay.confirmed,
                tier_label=review_state.tier_label,
                review_score=review_state.score,
                wave_count=review_state.recent_wave_count,
                wave_detected=wave_state.wave_detected,
                expression_label=review_state.concern_label,
                expression_context_strength=review_state.concern_strength,
                demo_override=demo_override,
                activity_label=(None if activity_prediction is None else activity_prediction.label),
                activity_confidence=(
                    None if activity_prediction is None else activity_prediction.confidence
                ),
            )
            snapshots.append(snapshot)
            if pose_result.track_id is not None:
                self.event_recorder.record(snapshot, timestamp, self.frame_count)

        stale_keys = _discard_stale_tracks(
            self.person_states,
            self.frame_count,
            self.config.stale_track_frames,
            active_keys,
        )
        for track_key in stale_keys:
            if self.activity_recognizer is not None:
                self.activity_recognizer.remove_track(track_key)
            self.event_recorder.end_track(track_key, timestamp, self.frame_count)

        self.last_snapshots = snapshots
        _draw_global_hud(
            annotated,
            fps=self._fps,
            person_count=len(snapshots),
            identity_enabled=self.identity_enabled,
            recent_events=self.event_recorder.recent_messages,
            demo_override_active=demo_override_active,
        )
        self.frame_count += 1
        self.timer.tick()
        return annotated

    def reset_tracking(self, **event_context: Any) -> None:
        self.person_states.clear()
        if self.activity_recognizer is not None:
            self.activity_recognizer.reset()
        self.last_snapshots = []
        self.frame_count = 0
        self._timestamp_origin = None
        self._last_frame_clock = None
        self._fps = 0.0
        reset_pose_tracking = getattr(self.pose_analyzer, "reset_tracking", None)
        if callable(reset_pose_tracking):
            reset_pose_tracking()
        if "source" in event_context:
            event_context["source"] = _source_for_logging(event_context["source"])
        self.event_recorder.reset_tracks(**event_context)

    def close(self) -> None:
        try:
            if self.activity_recognizer is not None:
                average_latency = self.activity_recognizer.average_inference_latency_ms
                if average_latency is not None:
                    logger.info(
                        "Activity MLP incremental latency %.3f ms (%s inferences; excludes YOLO)",
                        average_latency,
                        self.activity_recognizer.inference_count,
                    )
        finally:
            try:
                if self.emotion_analyzer is not None:
                    self.emotion_analyzer.close()
            finally:
                self.event_recorder.close()

    def _prepare_identity_matcher(self, supplied_matcher: Any) -> Any:
        if not self.config.identity_enabled:
            return None
        if supplied_matcher is not _AUTO:
            return supplied_matcher
        try:
            ensure_models(identity_model_specs(self.config), self.config)
            return OpenCVFaceIdentifier(
                detector_model_path=self.config.face_detector_model_path,
                recognizer_model_path=self.config.face_recognizer_model_path,
                cosine_threshold=self.config.identity_cosine_threshold,
                min_score_margin=self.config.identity_min_score_margin,
                min_face_size=self.config.identity_min_face_size,
                min_face_confidence=self.config.identity_min_face_confidence,
            )
        except (ModelUnavailableError, FileNotFoundError, cv2.error) as exc:
            logger.warning("Identity disabled: %s", exc)
            return None

    def _prepare_emotion_analyzer(self, supplied_analyzer: Any) -> Any:
        if not self.config.expression_enabled:
            return None
        if supplied_analyzer is not _AUTO:
            return supplied_analyzer
        try:
            from emotion import FaceEmotionAnalyzer

            ensure_model(emotion_model_spec(self.config), self.config)
            return FaceEmotionAnalyzer(
                self.config.emotion_face_model_path,
                classifier_model_path=self.config.emotion_classifier_model_path,
                min_display_confidence=self.config.min_expression_display_confidence,
            )
        except Exception as exc:
            logger.warning("Expression analysis disabled: %s", exc)
            logger.debug("Expression analyzer initialization failed", exc_info=True)
            return None

    def _prepare_activity_recognizer(self, supplied_recognizer: Any) -> Any:
        if supplied_recognizer is not _AUTO:
            return supplied_recognizer
        if self.config.activity_model == "none":
            logger.info("Activity recognition OFF")
            return None
        if self.config.activity_model != "mlp":
            raise ValueError(f"Unsupported activity model: {self.config.activity_model}")

        from activity_recognition.live import LiveMLPActivityRecognizer

        logger.info(
            "Loading live MLP activity checkpoint: %s",
            self.config.activity_checkpoint_path,
        )
        recognizer = LiveMLPActivityRecognizer(
            self.config.activity_checkpoint_path,
            sequence_length=self.config.activity_sequence_length,
            confidence_threshold=self.config.activity_confidence_threshold,
            inference_interval=self.config.activity_inference_interval,
            smoothing_window=self.config.activity_smoothing_window,
        )
        logger.info(
            "Activity recognition ON (MLP, threshold %.0f%%, interval %s frames)",
            self.config.activity_confidence_threshold * 100.0,
            self.config.activity_inference_interval,
        )
        return recognizer

    def _new_person_runtime(self) -> PersonRuntime:
        return PersonRuntime(
            wave_monitor=self.wave_monitor_type(),
            review_monitor=self.review_monitor_type(
                min_concern_confidence=self.config.min_concern_expression_confidence
            ),
            identity_consensus=IdentityConsensus(
                window_size=self.config.identity_consensus_window,
                required_matches=self.config.identity_required_matches,
                ttl_frames=self.config.identity_ttl_frames,
            ),
            emotion_smoother=(
                self.emotion_smoother_type(
                    window_seconds=self.config.expression_smoothing_seconds,
                    uncertainty_threshold=self.config.min_expression_display_confidence,
                )
                if self.emotion_smoother_type is not None
                else None
            ),
        )

    def _report_identity_mode(self) -> None:
        if self.identity_matcher is None:
            logger.info("Identity names OFF; monitoring will run anonymously")
        elif not self.known_identities:
            logger.info("Identity names OFF; no enrolled faces were found")
        else:
            names = {identity.name for identity in self.known_identities}
            logger.info(
                "Identity names ON (%s encodings for %s people)",
                len(self.known_identities),
                len(names),
            )

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_frame_clock is not None:
            instantaneous = 1.0 / max(now - self._last_frame_clock, 1e-6)
            self._fps = instantaneous if self._fps == 0 else self._fps * 0.9 + instantaneous * 0.1
        self._last_frame_clock = now


def run_detection(source: int | str = 0, config: AppConfig | None = None) -> bool:
    """Run live multi-person pose, expression, tracking, and identity overlays."""
    runtime_config = config or AppConfig.from_env()
    configure_logging(runtime_config.log_level, runtime_config.log_dir)
    capture = open_capture(source)
    if not capture.isOpened():
        logger.error("Could not open camera source %s", _source_for_logging(source))
        capture.release()
        return False

    processor: MonitoringProcessor | None = None
    successful = False
    frames_processed = 0
    try:
        clock = CaptureClock(
            capture,
            recorded=isinstance(source, str) and Path(source).is_file(),
        )
        processor = MonitoringProcessor(runtime_config)
        processor.reset_tracking(mode="live", source=str(source))
        logger.info("Detection running; press q in the camera window to quit")
        while True:
            ok, frame = capture.read()
            if not ok:
                successful = clock.recorded and frames_processed > 0
                if not successful:
                    logger.error("Failed to grab a camera frame")
                break

            annotated = processor.process_frame(frame, timestamp=clock.timestamp())
            frames_processed += 1
            cv2.imshow("Live Monitoring", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Detection stopped by operator")
                successful = True
                break
    except Exception:
        logger.exception("Monitoring could not start or continue")
        print("Monitoring stopped. See output/logs/monitoring.log for details.")
    finally:
        capture.release()
        try:
            if processor is not None:
                processor.close()
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    return successful


def _discard_stale_tracks(
    person_states: dict[int, PersonRuntime],
    frame_count: int,
    stale_track_frames: int = DEFAULT_CONFIG.stale_track_frames,
    active_keys: set[int] | None = None,
) -> list[int]:
    active_keys = active_keys or set()
    stale_keys = [
        track_key
        for track_key, runtime in person_states.items()
        if (
            (track_key < 0 and track_key not in active_keys)
            or frame_count - runtime.last_seen_frame > stale_track_frames
        )
    ]
    for track_key in stale_keys:
        del person_states[track_key]
    return stale_keys


def _resolve_identity_overlay(
    frame: Any,
    person_box: tuple[int, int, int, int],
    runtime: PersonRuntime,
    frame_number: int,
    identity_interval_frames: int,
    identity_matcher: OpenCVFaceIdentifier | None,
    known_identities: Sequence[KnownIdentity],
) -> IdentityOverlay:
    fallback_box = runtime.cached_emotion.box if runtime.cached_emotion is not None else person_box
    if identity_matcher is None or not known_identities:
        label = _label_with_emotion("Person", None, runtime.cached_emotion)
        return IdentityOverlay("Person", fallback_box, label)

    due = runtime.cached_face_box is None or frame_number % identity_interval_frames == 0
    if due:
        x1, y1, x2, y2 = person_box
        cropped_img = frame[y1:y2, x1:x2]
        detected_face = (
            None if cropped_img.size == 0 else identity_matcher.detect_largest_face(cropped_img)
        )
        if detected_face is None:
            runtime.cached_face_box = None
            runtime.identity_consensus.observe(None, frame_number)
        else:
            runtime.cached_face_box = offset_box(detected_face.box, x1, y1)
            runtime.last_face_seen_frame = frame_number
            match = None
            if identity_matcher.is_face_usable(detected_face):
                match = identity_matcher.identify(
                    cropped_img,
                    detected_face,
                    known_identities,
                )
            runtime.identity_consensus.observe(match, frame_number)

    decision = runtime.identity_consensus.current(frame_number)
    face_visible = (
        runtime.cached_face_box is not None
        and frame_number - runtime.last_face_seen_frame <= identity_interval_frames * 2
    )
    if not face_visible:
        label = _label_with_emotion("Person", None, runtime.cached_emotion)
        return IdentityOverlay("Person", fallback_box, label)

    name = decision.name if decision.confirmed else "Unknown"
    face_box = runtime.cached_face_box or fallback_box
    return IdentityOverlay(
        name=name,
        face_box=face_box,
        label_text=_label_with_emotion(name, decision.score, runtime.cached_emotion),
        score=decision.score,
        confirmed=decision.confirmed,
        face_visible=True,
    )


def _label_with_emotion(name: str, score: float | None, emotion: Any) -> str:
    label = name
    if score is not None and name not in {"Person", "Unknown"}:
        label = f"{name} {score:.2f}"
    if emotion is not None:
        label = f"{label} | {emotion.display_label} {emotion.confidence:.0%}"
    return label


def _source_for_logging(source: object) -> str:
    """Remove URL credentials and query parameters from operator logs."""
    value = str(source)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid source URL>"
    if not parsed.scheme or not parsed.netloc:
        return value

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def _draw_review_overlay(
    frame: Any,
    person_box: tuple[int, int, int, int],
    review_state: ReviewState,
    wave_detected: bool,
    context_activated: bool,
    activity_label: str | None = None,
    activity_confidence: float | None = None,
) -> None:
    px1, py1, px2, py2 = person_box
    cv2.rectangle(frame, (px1, py1), (px2, py2), review_state.color, 3)
    status_lines = [review_state.tier_label]
    if activity_label is not None and activity_confidence is not None:
        status_lines.insert(
            0,
            f"Activity: {activity_label.title()} {activity_confidence:.0%}",
        )
    status_lines.append(
        f"waves:{review_state.recent_wave_count} | context:{review_state.concern_strength:.0%}",
    )
    if review_state.concern_expression_active:
        status_lines.append(f"{review_state.concern_label} {review_state.concern_strength:.0%}")
    _draw_status_panel(
        frame,
        person_box,
        status_lines,
        review_state.color,
        avoid_global_hud=activity_label is not None,
    )

    if wave_detected:
        message = "Right-hand wave counted"
    elif context_activated:
        message = "Expression context active"
    else:
        return
    cv2.putText(
        frame,
        message,
        (px1, min(frame.shape[0] - 15, py2 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        review_state.color,
        2,
    )


def _draw_status_panel(
    frame: Any,
    person_box: tuple[int, int, int, int],
    lines: list[str],
    color: tuple[int, int, int],
    *,
    avoid_global_hud: bool = False,
) -> None:
    x1, y1, x2, y2 = person_box
    available_width = max(1, x2 - x1 - 8)
    line_height = 15
    panel_height = 5 + line_height * len(lines)
    panel_top = y1
    if (
        avoid_global_hud
        and x1 < min(frame.shape[1], GLOBAL_HUD_WIDTH)
        and y1 < GLOBAL_HUD_MAX_HEIGHT
    ):
        panel_top = min(
            GLOBAL_HUD_MAX_HEIGHT,
            max(y1, y2 - panel_height),
        )
    panel_bottom = min(y2, panel_top + panel_height)
    cv2.rectangle(
        frame,
        (x1, panel_top),
        (x2, panel_bottom),
        color,
        cv2.FILLED,
    )
    for index, text in enumerate(lines):
        preferred_scale = 0.48 if index == 0 else 0.40
        text_width = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            preferred_scale,
            1,
        )[0][0]
        scale = (
            preferred_scale
            if text_width <= available_width
            else preferred_scale * available_width / max(1, text_width)
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, panel_top + 13 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
        )


def _draw_identity_overlay(
    annotated: Any,
    identity_overlay: IdentityOverlay,
    review_color: tuple[int, int, int],
) -> None:
    _draw_label_box(
        annotated,
        identity_overlay.face_box,
        identity_overlay.label_text,
        review_color,
    )


def _draw_label_box(
    frame: Any,
    box: tuple[int, int, int, int],
    label_text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    frame_height, frame_width = frame.shape[:2]
    preferred_scale = 0.6
    thickness = 1
    text_width, text_height = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_DUPLEX,
        preferred_scale,
        thickness,
    )[0]
    max_width = max(80, frame_width - 12)
    scale = (
        preferred_scale
        if text_width + 12 <= max_width
        else preferred_scale * max_width / max(1, text_width + 12)
    )
    text_width, text_height = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        thickness,
    )[0]
    label_width = min(frame_width, text_width + 12)
    label_height = max(24, text_height + 14)
    label_x1 = min(max(0, x1), max(0, frame_width - label_width))
    label_y1 = y2 - label_height
    if label_y1 < 0:
        label_y1 = min(max(0, y1), max(0, frame_height - label_height))
    label_x2 = label_x1 + label_width
    label_y2 = min(frame_height, label_y1 + label_height)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
    cv2.putText(
        frame,
        label_text,
        (label_x1 + 6, label_y2 - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (255, 255, 255),
        thickness,
    )


def _draw_global_hud(
    frame: Any,
    fps: float,
    person_count: int,
    identity_enabled: bool,
    recent_events: tuple[str, ...],
    demo_override_active: bool,
) -> None:
    lines = [
        f"FPS {fps:4.1f} | people {person_count} | identity {'ON' if identity_enabled else 'OFF'}"
    ]
    lines.extend(recent_events[-2:])
    panel_width = min(frame.shape[1], GLOBAL_HUD_WIDTH)
    panel_height = 24 + 18 * (len(lines) - 1)
    cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (25, 25, 25), cv2.FILLED)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (8, 17 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )
    if demo_override_active:
        text = "DEMO OVERRIDE ACTIVE"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
        x1 = max(0, frame.shape[1] - text_width - 24)
        cv2.rectangle(frame, (x1, 0), (frame.shape[1], 32), HIGH_COLOR, cv2.FILLED)
        cv2.putText(
            frame,
            text,
            (x1 + 8, 23),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


if __name__ == "__main__":
    run_detection()
