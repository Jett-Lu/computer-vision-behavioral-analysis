# Visualization and video overlays

Phase 4 adds presentation controls without changing the Phase 3 detector,
ByteTrack configuration, recognition decisions, snapshots or JSONL payloads.
`MonitoringProcessor` completes inference and event generation before calling
`OverlayRenderer`. Both live frames and media exports use this same renderer.
No dashboard, additional recognition model or new dependency is introduced.

## Display semantics

- `Track 7` uses the exact assigned ByteTrack ID in the snapshot and JSONL event.
  An assigned ID is an established tracker association, never a verified real-world
  identity. `Track pending` means no ID has been assigned; internal negative
  placeholder keys are never shown. IDs may be reused after source resets.
- The track palette is deterministic and independent of demonstration review
  levels. Colors repeat after six IDs; numbers disambiguate tracks.
- Skeletons consume the same confidence-filtered landmarks as Phase 3. Missing
  keypoints produce no invented joints or connecting lines. Contrasting joint
  dots and outlined, antialiased connections improve visibility.
- Panels show `DEMO` review status, wave counts, optional activity and independently
  gated identity/expression labels. `Identity: unknown` means an observed face
  lacks confirmation; `unavailable` means there is no displayed prediction.
  The recent-event HUD reports cleared states explicitly when those events fit.
  Existing confirmation/expiry rules and expression uncertainty markers remain.
- Panels fit the frame and avoid each other. Long text ends in `...`, unsupported
  OpenCV font characters become `?`. On crowded or very small frames, lower-priority
  rows or entire panels may be omitted. The HUD keeps a compact row at 160x120;
  extremely small frames cannot fit text. Labels can cover image content.
- Anonymous pose-only mode omits identity/expression rows and corresponding HUD
  events. Turning off a display layer does **not** disable its inference or logging;
  continue using `--pose-only`, `--no-identity`, and `--no-expression` for privacy.
- Trails represent recent bottom-center image positions, not calibrated distance
  or speed. Each person has a bounded independent history. Phase 3 ambiguity,
  gaps and abrupt box discontinuities start fresh history; expiry/reset/source
  changes remove it. This inherits Phase 3's conservative attribution tradeoffs.

Behavioral review levels, expressions and gestures remain unvalidated demonstrations
and are not evidence of dangerous intent.

## Configuration

Environment variables work with the existing live, photo and video commands.
Programmatic callers can use `dataclasses.replace(config, overlays=OverlayConfig(...))`.
All settings below affect presentation only.

| Variable (`MONITOR_OVERLAY_` prefix) | Default | Meaning / accepted bounds |
| --- | --- | --- |
| `BOXES` | true | Person bounding boxes |
| `SKELETONS` | true | Pose connections and joints |
| `TRACK_IDS` | true | Numerical ID or pending label |
| `LABELS` | true | Demo review, identity and expression rows |
| `ACTIVITY` | true | Activity prediction / unavailable row |
| `TRAILS` | false | Movement trails |
| `HUD` | true | FPS, assigned/pending counts, recent events |
| `LINE_THICKNESS` | 2 | Geometry width, 1–8 pixels |
| `TEXT_SCALE` | 1.0 | Multiplier on resolution-aware text sizing, 0.5–2 |
| `PANEL_ALPHA` | 0.82 | Panel background opacity, 0–1; text remains opaque |
| `TRAIL_POINTS` | 30 | Maximum history samples per track, 2–300 |
| `TRAIL_SECONDS` | 2 | Maximum trail age, 0.1–30 seconds |

Booleans accept true/false, yes/no, on/off or 1/0. Invalid environment values use
defaults; numeric bounds clamp. To remove all overlays, disable all seven layers,
including `LABELS`. A hidden layer does not suppress related HUD event messages;
hide the HUD separately when needed. Defaults preserve existing content, with
additional numerical IDs and clearer unavailable/demo labels. The old private
drawing helpers in `detection.py` have been replaced; the public pose drawing API
is unchanged. Existing JSONL consumers require no Phase 4 schema changes.

## Synthetic samples and validation

Run from the checkout:

```powershell
python scripts/overlay_samples.py --output output/phase4-samples
```

This creates a 960x540 PNG, a 60-frame MP4, annotated versions and
`measurements.json`. All poses are scripted coordinates over a synthetic grid;
no models, identity assets, camera or personal media are used. The real photo/video
export paths and Phase 2 private-directory/atomic-output helpers are exercised.
Generated media stays in ignored runtime storage. The displayed FPS in these
samples is the synthetic processing loop, not measured camera/model throughput.

The renderer fixture is deterministic at a fixed FPS value; tests compare repeated
pixel output and panel geometry rather than brittle cross-platform font hashes.
Tests cover independent layers, tiny frames, overlapping labels, missing joints,
privacy redaction, ID text, bounded independent trails, discontinuities, source
reset/expiry, and annotated image/video decoding. Display-on/off runs produce
identical snapshots and JSONL payloads after excluding timestamps/session UUIDs.

A local Windows 11 / Python 3.12.14 / OpenCV 4.13.0 run measured 100 iterations
after 10 warmups, two people, 1280x720, 30-point trails:

| Operation | Median | 95th percentile |
| --- | ---: | ---: |
| Frame copy, overlays off | 0.47 ms | 0.61 ms |
| Frame copy plus all overlays | 1.83 ms | 2.10 ms |

These are renderer-only observations from one machine, excluding inference,
capture and video encoding. They are not an end-to-end FPS claim or a comparison
against Phase 3. Crowded label placement costs more as people and occupied panels
increase; this two-person measurement must not be generalized to crowded scenes.
No physical-camera or representative-video validation was available. Trail
continuity and identity correctness still have the documented Phase 3 limits.
