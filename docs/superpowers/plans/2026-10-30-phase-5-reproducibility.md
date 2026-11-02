# Phase 5 reproducibility implementation plan

Implement the approved Phase 5 scope from `27ded0813a804fb01539f84102e7ac18dff52ec9` on `codex/phase-5-ml-reproducibility` only. Reuse the clean checkout and existing experiment modules; do not change live inference, HMDB51 fold assignment, tracking, privacy, events or overlays.

1. Test then implement strict source/cache provenance validation, cross-partition byte-content duplicate detection, extraction configuration/weights/dependency fingerprints, and manifest-bound cache hashes. Preserve valid caches. Legacy unverified caches must be regenerated before new experiments; historical files remain readable for auditing.
2. Test then isolate Python/NumPy/Torch RNG and deterministic settings around each classifier run, using stable per-model seeds. Record full training options and provenance in checkpoints. Protect existing checkpoints/reports against accidental overwrite.
3. Test then add independent saved-prediction metric reconstruction with canonical labels, duplicate/incomplete partition detection, per-class counts and errors. Save exact partitions, checkpoint/cache hashes and environment in new experiment reports.
4. Test then add warmups/repetitions and mean/median/p95 measurements with distinct classifier-only, preprocessing, backbone, and cached-input pipeline boundaries. Mark decoding/pose/full-video latency unavailable unless actually measured; do not derive them from cache extraction timestamps.
5. Inspect model limitations and historical documentation; write an evidence-based experimental plan without retraining. Run all tests, lint/type/dependency/build/wheel checks, independent review, then push and verify six CI jobs. No merge or Phase 6.

Local inventory: detector assets exist, but no HMDB51 data, activity cache, activity checkpoints or saved predictions were found in the configured default directories. No full benchmark or model retraining is authorized by this implementation plan; tiny synthetic CPU regression training is test-only.

## Review record

One independent whole-diff review found three important issues. Each was reproduced
by a failing test before correction: selected CUDA/MPS timing synchronization, MPS
RNG restoration on success/failure, and source timing's repeated predictor setup.
The timing contract now names predictor initialization/backend warmup on every clip;
production tracking resets remain unchanged.

Two minor follow-ups are deferred: include the supplied manifest digest in the
standalone reconstruction report (the evaluation run manifest already records it),
and make a mock extractor factory consume RNG in the full cold/warm integration
test (a separate training regression already tests prior RNG consumption).

Decisions: retain strict regeneration requirements for unverified experimental
artifacts; this costs migration work but prevents unsupported provenance claims.
Preserve per-clip predictor setup in source timing to match actual extraction;
these timings must not be interpreted as warmed-predictor-only inference.
Actual dataset accuracy, historical metric reconstruction, physical camera tests
and GPU reproducibility remain unverified without the missing inputs/hardware;
synthetic tests cannot substitute for those experiments.
