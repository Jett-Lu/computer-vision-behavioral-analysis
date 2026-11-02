# ML experiment reproducibility and evaluation

Phase 5 hardens offline experiments. It does not change the live recognition,
tracking, identity, privacy, event or overlay policies, and introduces no new
recognition architecture. No HMDB51 videos, activity checkpoints or saved
prediction CSVs were available locally during this phase. No new dataset
accuracy, real-model latency, GPU reproducibility or camera result is claimed.
Synthetic regression training and timing validate the infrastructure only.

## Evidence and compatibility

The tables in [the historical results](activity_recognition_results.md) and the
README are preserved. Their underlying predictions/checkpoints are unavailable
here, so their metrics and reported confusion cases have not been independently
reconstructed. Their uniform frame sampling differs from the current fixed
source-time sampling. The README and historical report also contain different
latency numbers; do not treat those as a single reproducible run or resolve the
discrepancy by choosing the more favorable figure.

New experiments require provenance-bearing caches. Old manifests can still be
read for historical metric reconstruction, but training/evaluation reject legacy
caches and checkpoints that lack the new evidence. Regenerate into a new cache
directory, and train only under an approved experiment plan. The live MLP loader
continues to accept its existing checkpoint format; it ignores the additional
offline provenance fields. Historical checkpoints/reports are never overwritten
by the training or evaluation entry points: choose fresh output directories.
Preparation can regenerate invalid caches, so use a separate cache directory to
retain earlier experiments. Evaluation requires the exact training manifest,
source videos and matching caches; relocating/editing a manifest requires a new
experiment. This is intentionally stricter than historical evaluation.

## Dataset and cache contracts

Preparation keeps the original HMDB51 fold definitions and deterministic
validation selection unchanged. It records the selected official split-file
hashes, or the two input mirror-CSV hashes for fold 1. It records every source
identifier, video SHA-256, class/index, partition, cache path and cache-file
SHA-256 in the manifest. These hashes establish consistency with the operator's
inputs, not certification that a third-party mirror is official or authentic.

Before training or evaluation, all partitions are checked, including the test
partition when training. This is a leakage/integrity audit only: test features
and predictions are not used in fitting, standardization or checkpoint selection.
Repeated byte-identical source content across partitions is rejected even under
different filenames. This does not detect transcoded copies, related scenes,
shared actors or near duplicates. Within-partition duplicates are permitted;
inspect the recorded source hashes before interpreting effective sample size.

Each pose/crop cache binds its key, source, class and partition to its extraction
contract, checks shape/dtype/finiteness/range, and hashes the pose and crop arrays.
Swapping caches, modifying valid-looking values, dropping manifest records or
changing source content is rejected. Cache reuse also checks:

- Actual YOLO weights and tracker YAML hashes; detection, display/pose and
  keypoint-confidence thresholds; IoU; relevant implementation files.
- Frame count, fixed sampling interval and sampling contract version; crop size,
  padding, RGB conversion, resize method, largest-person and missing-pose rules.
- Python, NumPy, Torch, Torchvision, Ultralytics and OpenCV versions.
- For frozen features: source NPZ hash, sample/label/partition, actual pretrained
  weight-file hash, exact weight enum, transforms, aggregation and implementation
  hashes, Torch/Torchvision versions and vector digest/shape/finiteness.

Unchanged valid caches are reused without constructing models. These conservative
fingerprints can invalidate caches after a source-file or dependency change even
if its numerical effect is harmless. File hashes are not a defense against an
attacker rewriting both artifacts and their manifests. Use trusted local assets
and retain the immutable experiment artifacts together. Concurrent writers to
the same experiment directory are unsupported.

Pose weights must already exist at the configured path before preparation can
fingerprint them. Provision the approved asset first (the normal model asset
workflow remains available); preparation will not silently download a different
pose model just to inspect caches. Frozen Torchvision feature preparation can
download the explicitly selected pretrained weights if absent. Evaluation needs
the same local backbone weight files. The code does not download datasets unless
the existing explicit `--download-videos` option is used.

Data, model caches and reports belong under ignored `data/hmdb51/`,
`data/activity_cache/`, `data/activity_models/` and `output/` paths. Manifest and
error reports can contain local paths and video names: review them before sharing.
Do not commit licensed/private videos, crops, checkpoints or private metadata.
Existing private directory, atomic replacement, symlink/junction and hardlink
helpers remain in use; no protections or retention settings are relaxed.

## Randomness and training record

Each model uses a seed derived from SHA-256 of `root-seed:model-name`. The
classifier construction, dropout and training run execute inside an isolated
Python/NumPy/Torch CPU/CUDA/MPS RNG context. The data-loader shuffle generator is
local and uses the recorded root seed. Previous model construction and a cold
feature cache cannot advance the classifier's initialization stream. Random
state and determinism flags are restored even on failure.

During training, deterministic Torch algorithms are requested in warning mode,
cuDNN benchmarking is disabled and cuDNN deterministic behavior is enabled.
Warnings must be retained when an operation cannot provide deterministic behavior.
Checkpoints record root/model seeds, epochs, patience, batch size, learning rate,
optimizer/loss/selection policy, training-only standardization, exact dataset
manifest hash and content, backbone contracts and environment. Environment records
include dependency versions, platform/processor, CPU threads, device/GPU,
CUDA/cuDNN and CUBLAS workspace configuration. They do not promise bit-identical
GPU results across hardware, drivers, releases or nondeterministic operations.
This follows the limits described in
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Running an approved experiment

These commands are an example for a separately approved future run, not a record
of training performed in Phase 5. Use a unique run ID and retain the source inputs.
Install `requirements.txt` and `requirements-activity.txt`, then save the resolved
dependency environment alongside the reports. From an activated virtualenv:

```powershell
python scripts/prepare_activity_data.py --dataset-root data/hmdb51/hmdb51_org --annotations-root data/hmdb51/splits --cache-dir data/activity_cache/run-001 --fold 1 --seed 2026 --frames 16 --sampling-interval-seconds 0.1
python scripts/train_activity_models.py --manifest data/activity_cache/run-001/manifest.json --feature-dir data/activity_cache/run-001/features --model-dir data/activity_models/run-001 --results-dir output/activity-run-001 --seed 2026 --epochs 20 --patience 4 --batch-size 32 --learning-rate 0.001 --device cpu --skip-evaluation
python scripts/evaluate_activity_models.py --manifest data/activity_cache/run-001/manifest.json --feature-dir data/activity_cache/run-001/features --model-dir data/activity_models/run-001 --results-dir output/activity-run-001 --device cpu --latency-samples 20 --latency-warmup 3 --latency-repeats 10
python scripts/reconstruct_activity_metrics.py output/activity-run-001/predictions.csv --manifest data/activity_cache/run-001/manifest.json --output-dir output/activity-run-001-reconstruction
```

The reconstruction command needs neither videos nor model inference. It verifies
test membership/labels against the supplied manifest, rejects duplicate/missing
predictions and inconsistent model sample sets, then independently computes
accuracy, macro precision/recall/F1, confusion matrices, class support and an
error list. Without `--manifest`, membership is explicitly marked unverified.
It cannot establish that a saved prediction was actually produced by its claimed
checkpoint. Evaluation saves both the original aggregate metrics and independently
reconstructed metrics, the predictions digest, exact checkpoint hashes, training
configuration and dataset quality counts by class/partition. Missing-keypoint
fractions describe cached visibility, not YOLO accuracy.

## Latency boundaries

`metrics.json` contains raw repeated timings plus count, mean, median and p95;
`latency.csv` has one row for each actually measured boundary. The legacy
`inference_latency_ms` field remains the cached-input pipeline mean, whose scope
is stated explicitly. Defaults use 3 warmups and 10 measured repetitions per
selected test video, batch size one video; MobileNet's frame batch is the clip's
frame count. The first N test records in manifest order are selected and recorded;
this is a reproducible convenience sample, not an unbiased latency sample.

| Boundary | Included | Excluded |
|---|---|---|
| `classifier_only` | Head forward on device-resident standardized features | All feature extraction/preprocessing |
| `backbone_only` | Frozen extractor forward and temporal pooling | Head and input preprocessing; unavailable for MLP |
| `preprocessing` | Tensor conversion/normalization, transfer, feature standardization | Video decode, person crop and pose extraction |
| `cached_input_pipeline` | Preprocessing, backbone when applicable, head, synchronization | Cache I/O, video decode, YOLO and crop construction |
| Optional `pose_extraction` | YOLO predictor initialization/backend warmup per clip, then tracking on all sampled frames | Decode and classification |
| Optional `end_to_end_video` | Decode/sample window, YOLO predictor initialization/warmup, tracking, person selection, crop/pose normalization, tensors, backbone/head | Initial weight loading, reset call, rendering, encoding, live UI |

Pass `--include-source-latency` to additionally measure decoding, pose extraction,
crop/pose normalization and the full offline source-to-classification path. This
requires local videos/assets and an extraction configuration matching the manifest;
the pose model is explicitly moved to the classifier's selected device. Otherwise
pose/full-source timings are null, not zero or estimates from cache timestamps.
The selected CUDA device or MPS is synchronized at stage boundaries. Repetitions
reuse filesystem caches and loaded weights. Source measurements deliberately
match extraction's per-clip tracking reset: each timed clip recreates the YOLO
predictor, including backend setup and warmup. Discarded warmup repetitions do
not remove that per-clip cost. These are neither fully warmed predictor timings
nor application cold-start or live multi-person FPS. Do not compare cached-input means with full-source means as if
their boundaries matched. No such real-model measurements were run in Phase 5.

## Evidence-based limitations and next experiment plan

| Area | Implementation evidence | Consequence and proposed controlled experiment |
|---|---|---|
| MLP | Flattened frame/17-keypoint/x-y-visibility vector; 128-unit hidden layer; missing triples zeroed | Translation/scale information is intentionally lost; missing joints can correlate with a class. Report metrics by visibility and test a masking/quality-gating ablation before changing the model. |
| MobileNetV2 | Frozen ImageNet V2 extractor with frame-mean aggregation and linear head | Frame order is discarded, making walking/running discrimination dependent on appearance/pose cues. First compare error cases and sampling windows without a new architecture. |
| S3D | Frozen Kinetics-400 V1 extractor; current default 16-frame, 0.1-second source-time grid | Temporal coverage and crop geometry differ from its published pretrained evaluation recipe. Compare sampling/cropping ablations on validation, with a separately fingerprinted cache per contract. |
| Person selection | Largest detected person selected independently each frame; full-frame crop if none | Bystanders or crossings can contaminate a labeled action sequence. Manually review consented error clips and visibility strata before an actor-continuity ablation. |
| Labels/imbalance | Four HMDB actions; cross-entropy with no weighting; clip-level labels | Clip labels need not describe every frame. Inspect actual partition/class counts and uncertainty; apply weighting only if supported by measured imbalance. Historical balanced counts cannot establish new-run balance. |
| Unknown handling | Offline evaluation always selects one of four classes; live MLP uses a confidence threshold and smoothing | Closed-set accuracy says nothing about rejection of unrelated activities or calibrated probability. Collect a consented unknown/OOD set; tune rejection/calibration only on validation and report coverage/error separately. |

Torchvision documents S3D's pretrained evaluation with a longer temporal window
and a resize/center-crop transform; this repository uses person crops. That is a
known protocol difference, not evidence that any particular accuracy improvement
will result from changing it. See the
[official S3D weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.s3d.html).

Before new training: acquire permitted data/checkpoints; audit original fold files,
content duplication and class/visibility distributions; reconstruct historical
predictions if recovered. Freeze a baseline protocol and predeclare root seeds
and official folds. Use validation for head selection, any weighting, sampling
ablations and unknown thresholds. Evaluate the untouched test partition once for
each frozen protocol; report per-class metrics, seed/fold variability and paired
errors with identical source sets. Measure latency under matched boundaries and
hardware. Record each ablation in its own directories. Do not silently revise the
official split to bypass duplicate-content failures: investigate the inputs first.

Activity, gesture and expression outputs remain demonstration information. None
of these experiments validates dangerous intent, threat prediction or automated
decisions about people.
