# Tracking and temporal attribution

Phase 3 builds on Phase 2 commit `b431726`. Identity and expression switches,
anonymous pose-only mode, secure storage, retention, and enrollment validation
remain in effect. Review levels are demonstration rules, not validated risk
predictions. No facial expression or gesture establishes intent.

## ByteTrack thresholds

`src/monitoring_assets/bytetrack.yaml` is included in installed wheels. The
application explicitly selects it instead of relying on library defaults.

| Setting | Value | Purpose |
| --- | --- | --- |
| `MONITOR_POSE_DETECTION_CONFIDENCE` | 0.10 | Candidate admission before association; environment values clamp to 0.01–0.10 |
| `track_high_thresh` | 0.25 | First association stage |
| `track_low_thresh` | 0.10 | Lower bound for recovery of existing tracks |
| `new_track_thresh` | 0.30 | Prevent weak candidates from starting tracks |
| `track_buffer` | 30 | Lost-track buffer in tracker frames, not guaranteed wall-clock seconds |
| `match_thresh` | 0.80 | First-stage matching threshold |
| `fuse_score` | true | Fuse detection confidence with association cost |
| `MONITOR_POSE_CONFIDENCE` | 0.30 | Existing setting now filters displayed/analysed results after association |

Previously the 0.30 detector cutoff prevented the low-confidence association
stage from receiving candidates. Weak candidates now maintain internal tracking
state without producing low-confidence overlays or behavioral observations.
YOLO NMS IoU remains separately configurable. See
[Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track).

The deterministic installed-ByteTrack fixture retains one ID through three
0.15-confidence frames and three empty frames, then recovers it at 0.90. A
spatially separate 0.15 candidate creates no extra track. This is a synthetic
regression result, not a measured improvement in IDF1, HOTA, MOTA, or field accuracy.
No physical-camera or representative-video validation was performed.

## Attribution rules and tradeoffs

Missing or unusable faces do not count as rejected recognition matches and do
not renew consensus. Confirmed and pending evidence expire. An explicit
rejection or contradictory identity clears old votes; confirmation requires
fresh agreement. More than one face in a body crop, or a face center inside
another body, invalidates attribution. Cached face ownership is checked even
between scheduled recognition frames.

Body overlap of at least 25% of the smaller box suppresses identity, expression,
and activity attribution. A gap in displayed observations or non-overlapping
successive body boxes starts fresh per-person temporal state. Tracking IDs may
survive these gaps, but names, gesture histories and activity windows do not
automatically follow them. This favors false negatives over transferring
someone else's state. Fast movement and harmless overlap can reset state.

ByteTrack IDs are temporary association keys, not identity guarantees. A
visually continuous ID switch without detectable ambiguity can still evade
these safeguards, particularly between face-recognition frames. There is no
claim of perfect identity attribution.

## Right-hand waves

Horizontal motion is relative to the right shoulder and normalized by shoulder
width with frame aspect-ratio correction. The movement threshold is 0.25
shoulder widths; the raised-hand margin is 0.15 widths. Both shoulders and the
right wrist must have usable keypoints. Missing/degenerate keypoints,
non-monotonic timestamps, or observation gaps over 0.5 seconds break the motion
sequence. Rolling counts and cooldown remain in effect.

Synthetic equivalent gestures count once at four body scales; body translation,
small jitter, missing scale and interrupted motion are covered by tests. These
thresholds need representative-video calibration. Side-on bodies with poorly
resolved shoulder width may be rejected. Left-hand support is unchanged.
Direct callers of `RightHandWaveMonitor` should provide frame shape for correct
aspect ratio; custom movement/margin values now use body-scale units.

## JSONL lifecycle and replay

Existing top-level fields and event names retain their meanings. New fields are
additive: `schema_version: 2`, `session_id`, monotonic report `sequence`, and
`state`. New event types include `identity_cleared`, `expression_cleared`,
`activity_unavailable`, `track_unavailable`, `track_available`, and `state_updated`.
`track_ended` includes a reason: `tracker_terminated`, `source_reset`, or `shutdown`.
Shutdown timestamps use the latest processed source frame, including empty frames.

Replay records in sequence order, keyed by `(session_id, track_key)`, replacing
each track's state with the event's `state`. Render only `visible: true`; remove
ended tracks. Source resets terminate the old session before starting a new ID.
Null activity/expression values mean unavailable, never carry-forward
predictions. State includes score-only changes and the actual displayed
expression label/confidence separately from the legacy review-context expression
fields. It reconstructs meaningful per-track labels and review state, not video
pixels, skeleton geometry, FPS, or the global recent-message panel.

Disabled identity/expression fields are omitted from both legacy fields and
state. Consumers that ignore unknown keys/events remain compatible; consumers
with strict schemas or exhaustive event enumerations must accept these additions.
Full state events may increase report volume; existing retention still applies.
