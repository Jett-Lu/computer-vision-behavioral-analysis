# Phase 2 Privacy and Security Implementation Plan

**Goal:** Implement the approved Phase 2 scope while preserving verified Phase 1 commit c3db84e.

**Architecture:** Extend existing data-policy helpers and reuse them at managed write boundaries. Keep the current pipeline and recognition defaults; gate optional model initialization and reporting independently.

**Constraints:** Work only on codex/phase-2-privacy-security. No merge, Phase 3, weakened static checks, or weakened retention/ACL protections. Behavioral output remains an unvalidated demonstration.

- [x] Reproduce diagnostics permission/probe and event hardlink defects with tests, then add secure append and atomic-output helpers in data_policy.py.
- [x] Apply helpers to exports, enrollment, logs, models, expression caches, validation reports, and benchmark outputs. Preserve temporary suffixes for codecs and existing output on failure.
- [x] Add independent identity_enabled/expression_enabled configuration, CLI opt-outs, and pose-only mode. Verify model preparation skips disabled modules and event output omits identity data.
- [x] Reproduce fail-open enrollment, then reject unavailable/failed validation. Allow unchecked fallback only through an explicit CLI/API action; disable enrollment in anonymous mode.
- [x] Complete cross-platform safety, compatibility, and disabled-module regressions; review the whole diff.
- [x] Run unittest, Ruff lint/format, Pyright, pip check, compileall, build, external wheel verification.
- [ ] Commit/push only to Phase 2 and verify all six CI jobs; report limitations and stop.

Review focus: original linked path spelling must survive until validation; temporary output must not overwrite another process's fixed filename; disabled modules must not consume injected models or download assets; failed native ACL setup must leave no apparently initialized directory; file rotation/replacement must validate destination and source.

Security boundary: these local-account controls are not encryption or a sandbox against the account owner/admin. Existing broad directory permissions require the explicit secure-data migration; new sensitive files are restricted before content is written. Cross-platform path-based native codecs still require trusted directories during an operation.

Local verification: 169 tests, 163 passed and 6 platform/privilege skips; Ruff, Pyright, pip check, compileall, isolated build, and external wheel verification passed. Independent review findings (capture cleanup and injected recorder privacy) fixed with failing-then-passing regressions. Hosted CI remains the final gate.
