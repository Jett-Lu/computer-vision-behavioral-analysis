# Activity Recognition Results

Historical report, preserved without updating its figures. Phase 5 did not have
the original videos, checkpoints or prediction files to reconstruct this run.
Current sampling and evidence requirements differ; see
[ML reproducibility and evaluation](ml-reproducibility.md) before making comparisons.

## Objective

This benchmark compares three compact approaches for recognizing walking,
running, standing, and sitting while keeping the existing monitoring pipeline
unchanged. Training and evaluation are offline. The resulting MLP checkpoint
can also be enabled as an informational per-person label in the live
application; MobileNetV2 and S3D remain offline comparison models.

## Dataset And Partitions

The data is the `walk`, `run`, `stand`, and `sit` subset of HMDB51 official
fold 1. The canonical output order is:

1. `walking`
2. `running`
3. `standing`
4. `sitting`

The official training videos were split deterministically at the video level.
All derived frames, crops, pose sequences, and frozen features retain the
partition of their source video.

| Partition | Videos per class | Total videos |
|---|---:|---:|
| Training | 56 | 224 |
| Validation | 14 | 56 |
| Official test | 30 | 120 |

The official test videos were used only for final inference and metric
calculation. They were not used for classifier fitting, early stopping,
threshold selection, feature selection, or preprocessing design. Frozen
backbone features were computed after partitions were established using fixed
pretrained weights and fixed transforms.

## Preprocessing

All approaches uniformly sample 16 source-order frames per video. YOLO Pose
selects the largest detected person and supplies an 8%-padded person crop.

- **MLP:** 17 YOLO keypoints per frame are represented as bounding-box-relative
  x/y coordinates plus visibility. The 16 x 17 x 3 sequence is flattened.
- **MobileNetV2:** each 224 x 224 RGB person crop is ImageNet-normalized. Frozen
  MobileNetV2 produces a 1,280-value vector per frame; 16 vectors are averaged.
- **S3D:** the 16 RGB crops form one normalized 3 x 16 x 224 x 224 clip. Frozen
  Kinetics-400 S3D produces one 1,024-value vector.

Preprocessing is identical for training and evaluation. YOLO outputs,
person crops, and frozen backbone features are cached per source video.

## Models And Training

- **MLP:** 816 inputs, a 128-unit ReLU hidden layer, dropout, and four outputs.
- **MobileNetV2:** frozen ImageNet-pretrained backbone, dropout, and a four-class
  linear head.
- **S3D:** frozen Kinetics-400-pretrained video backbone, dropout, and a
  four-class linear head.

Only the classification heads are trained for the MobileNetV2 and S3D
approaches. Features are standardized using training-partition statistics.
Training uses Adam, cross-entropy loss, a fixed seed of 2026, a maximum of 20
epochs, early stopping, and checkpoint selection by validation macro F1 with
validation loss as a tie-breaker.

## Comparison

These values were calculated from predictions for the 120 official test
videos. Precision, recall, and F1 use unweighted macro averaging across the
four classes.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Head training time | Measured latency |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 46.67% | 49.09% | 46.67% | 47.00% | 0.219 s | 0.773 ms |
| MobileNetV2 | 47.50% | 47.62% | 47.50% | 46.02% | 0.111 s | 189.605 ms |
| S3D | 42.50% | 42.14% | 42.50% | 41.35% | 0.065 s | 572.012 ms |

MobileNetV2 achieved the highest accuracy. The MLP achieved the highest macro
F1. S3D did not outperform the simpler approaches in this experiment.
Increased model complexity did not guarantee improved performance.

## Confusion Matrices

Matrix rows are actual classes and columns are predicted classes in the
canonical label order.

- **MLP:** the largest confusion was running predicted as walking, affecting
  11 of 30 running videos. Walking was predicted as running for 10 videos.
  Standing and sitting were each predicted as walking for 8 videos.
- **MobileNetV2:** the largest confusion was sitting predicted as standing,
  affecting 13 of 30 sitting videos. Walking was predicted as running for 9
  videos, and standing was predicted as running for 9 videos.
- **S3D:** the largest confusion was sitting predicted as standing, affecting
  15 of 30 sitting videos. Standing was predicted as sitting for 13 videos.
  Walking was split equally between running and sitting as its most frequent
  errors, with 9 videos in each direction.

These counts describe observed errors only; they do not establish their cause.

## Latency Boundaries

The latency values are not directly equivalent because their timed boundaries
are different. Cache loading and video decoding are excluded for every model.

- **MLP:** classifier inference from cached YOLO pose features only. YOLO pose
  estimation is excluded.
- **MobileNetV2:** cached-crop tensor conversion and normalization, frozen
  MobileNetV2 inference over 16 frames, feature averaging, and classifier
  inference.
- **S3D:** cached-clip tensor conversion and normalization, frozen S3D
  inference, and classifier inference.

No separate end-to-end latency is reported. A defensible end-to-end comparison
would need to include video decoding, YOLO pose estimation, person selection,
crop creation, model preprocessing, backbone inference, and classifier
inference under the same sampling and hardware conditions.

## Limitations

The results are limited by frozen pretrained backbones, limited four-class
adaptation, variation in HMDB51 sources, ambiguity between some activities,
short clips, and limited classifier-head training. The person-selection rule
uses the largest detected pose and may select a bystander in multi-person
clips. The MLP has no explicit temporal model beyond flattening the sampled
sequence. MobileNetV2 averages frame features and therefore discards temporal
ordering. S3D receives temporal input, but its frozen Kinetics-400 features were
not adapted beyond the final head.

Three different running clips produced the same all-zero MLP pose sequence
because YOLO Pose found no person in any sampled frame. Two clips are in the
training partition and one is in the official test partition. Their source
videos and cache artifacts are distinct, so this is not reuse across
partitions, but it is a pose-detection failure and an information-loss
limitation for the MLP.

None of these models should be considered production-ready from this
experiment.

The comparison is closed-set: evaluation selects the largest classifier logit
and does not apply confidence calibration, softmax-based acceptance, or an
unknown-activity threshold. Torchvision also marks the S3D video-model API as
beta, so future Torchvision upgrades require compatibility testing.

The optional live MLP applies a configurable threshold to smoothed softmax
probabilities and reports lower-confidence outputs as `unknown`. That policy
reduces label flicker but does not calibrate the classifier. Live activity
labels remain separate from review-tier scoring.

## Live MLP Incremental Latency

The live integration was measured on the same Windows CPU environment using
the saved MLP checkpoint, a 16-pose sequence, a five-frame inference interval,
and a five-result smoothing window. Across 516 tracked-person updates and 101
scheduled inferences, the mean incremental inference path took `0.245 ms`.
Amortized across every tracked-person update, activity handling took
`0.146 ms` per frame.

This measurement starts with pose landmarks already supplied by the existing
YOLO tracking stage. It includes bounding-box pose normalization, history
management, feature standardization, MLP inference, softmax, thresholding, and
smoothing. It excludes YOLO, frame decoding, identity and expression models,
event I/O, and display rendering, so it is not an end-to-end camera FPS result.

## Reproducibility

The measured run used Python 3.12.13, NumPy 2.4.4, PyTorch 2.12.0 CPU,
Torchvision 0.27.0 CPU, 16 frames per video, and seed 2026. The generated
`run_manifest.json` stores library versions, split counts, label mapping,
preprocessing boundaries, the dataset-manifest hash, and checkpoint hashes.
`predictions.csv` stores the expected and predicted label for every model and
official test video, allowing the metrics and confusion matrices to be
recalculated independently.

The validation pass confirmed that video keys, resolved paths, cache paths, and
binary video hashes do not cross partitions. Cache metadata matches its source
manifest row, and every frozen feature file is keyed to exactly one video.
Hashes of 6,394 unique cached crop frames, all 400 MobileNetV2 vectors, and all
400 S3D vectors had no cross-partition matches. The all-zero pose fallback
described above was the only cross-partition value collision among complete
pose sequences. Each checkpoint's feature mean and standard deviation were
independently recalculated from training features only. Metrics reconstructed
directly from the 360 saved prediction rows match `metrics.json` and
`model_comparison.csv` within `1e-12`.
