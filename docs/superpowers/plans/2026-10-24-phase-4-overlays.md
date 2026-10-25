# Phase 4 overlay implementation plan

**Goal:** Implement the user's approved Phase 4 scope on `codex/phase-4-visualization-overlays`, based on `60e25e0`, preserving inference, privacy, attribution and JSONL behavior.

**Architecture:** Keep MonitoringProcessor as the owner of model execution and per-person state. A small OpenCV renderer consumes completed snapshots and filtered poses after event generation. Configuration and trail history have no influence on predictions. No new models, dashboard, or dependencies.

**Decisions:** Reuse the existing clean checkout on a dedicated branch. Default trails off; other existing layers on, numerical IDs on. A missing ByteTrack ID is shown as `Track pending`; assigned IDs are tracker-established, never real-world identity. Use a deterministic per-ID palette, dark translucent panels, frame-aware sizing, and non-overlapping panel allocation. If space runs out, omit lower-priority text instead of drawing over other labels. Trail positions are normalized image coordinates, capped at 30 points/2 seconds by default; gaps and ambiguity reset them using Phase 3 state boundaries.

## Tasks and acceptance

1. Add OverlayConfig and a pure renderer in `src/overlays.py`. Test configuration bounds, independent layers, ID/pending text, privacy filtering, fitted panels, deterministic pixels, tiny frames, incomplete skeletons, and crowded labels before implementation. Package the new module.
2. Integrate rendering after snapshots/events in `src/detection.py`; keep old public pose drawing API. Add bounded trail histories to PersonRuntime. Test enabled/disabled settings produce identical snapshots and normalized events, trails remain independent and clear on gaps, ambiguity, teleports, expiry, reset and source change. Keep source images unchanged.
3. Exercise real annotated image/video export with synthetic poses and no model loading. Add a reproducible sample/benchmark script using existing secure output helpers, generate synthetic media and inspect it. Measure renderer overhead separately from inference; no camera FPS claims.
4. Run full unittest suite, Ruff check/format, Pyright, pip check, compileall, build and outside-checkout wheel verification. Obtain one independent whole-diff review, resolve significant findings with regression tests, commit and push only this branch, verify all six CI combinations. No merge or Phase 5.

## Review focus

- Display toggles must not change events or inference calls.
- Privacy must cover per-person labels and recent HUD messages.
- Tiny frames and exhausted label space must not draw invalid rectangles or overlap labels.
- Trailing paths must never bridge missing/ambiguous track intervals.
- Export/sample paths retain Phase 2 filesystem protections.

## Progress

- Phase 3 baseline: 187 tests, successful with six platform-specific skips.
- Renderer/configuration: tests failed before implementation and now pass. Added shared renderer and removed superseded private drawing helpers.
- Processor/trails: regression tests failed first; display-on/off events and snapshots now match. Bounded trails reset through existing Phase 3 lifecycle.
- Synthetic exports: real PNG/MP4 export and decode checks pass; samples visually inspected. Corrected bottom-edge label placement and missing compact HUD with failing regressions first.
- Whole-branch review: display/inference separation confirmed; crowded-layout performance limitation documented. Final suite/build/wheel/CI verification follows.
- Review fixes: retained compact HUD at 160x120 and confirmed-identity match scores, each with a regression observed failing first. No architectural expansion.
- Local final checks: 203 tests, six platform-specific skips; Ruff check/format, Pyright, pip check, compileall and distributions pass.
