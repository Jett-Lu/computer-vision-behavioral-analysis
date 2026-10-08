# Real-Time Computer Vision & Behavioral Analysis System

A real-time computer vision system for multi-person tracking, pose estimation,
activity recognition, and temporal behavioral analysis, with optional identity
recognition and facial-expression context. Each person is tracked using independent
temporal state, allowing movement and gesture patterns to be evaluated over time.

The project was developed as an applied artificial intelligence portfolio
piece to explore how multiple computer vision models can be integrated into a
single processing pipeline.

See [runtime configuration](docs/configuration.md) for environment variables,
filesystem requirements and operational limits, and the
[data access and retention policy](docs/data-governance.md) for protecting
recordings and identity data. See the
[final production audit](docs/production-audit.md) for verification results
and remaining release blockers.

<img width="1069" height="598" alt="image" src="https://github.com/user-attachments/assets/98c651a0-158e-46c2-81f4-dbaf38517503" />

## Project Overview

The system processes live or recorded video and maintains an independent state
for every tracked person.

For each detected person, the application can display:

* a persistent tracking ID
* pose landmarks and skeleton rendering
* enrolled-person identity
* facial-expression context
* human activity classification
* completed right-hand wave count
* current review level
* recent state changes

The application combines spatial model outputs with temporal logic. Individual
detections are not treated as isolated frames. Instead, tracking history,
gesture events, identity agreement, activity sequences, and expression
smoothing are maintained separately for each person.

```text
                    Camera / Recorded Media
                              |
                              v
                     OpenCV Frame Handling
                              |
                              v
                         YOLOv8 Pose
                    People + Body Keypoints
                              |
                              v
                          ByteTrack
                         Tracking IDs
                              |
                              v
                 Independent State per Track
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    GESTURE ANALYSIS     FACE ANALYSIS      ACTIVITY ANALYSIS
          |                   |                (optional)
          v                   v                   |
    Raised Right-Hand   Optional Identity         v
    Wave Detection      YuNet + SFace       Pose History
          |                   |              16 Frames
          v                   v                   |
    Rolling Window      Facial-Expression        v
    Wave Counter        Estimation          MLP Classifier
          |                   |                   |
          v                   v                   v
    Rule-Based          Temporal            Smoothed
    Review Level        Smoothing           Activity Label
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                  Combine Per-Person Results
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Annotated Display       Structured Events
            / Media Export          / JSONL Logging
```

Face and activity predictions provide context only.
Review levels are determined by repeated-wave rules.
## Key Features

### Multi-Person Detection and Tracking

YOLO Pose and ByteTrack are used to detect and track multiple people.

Each person receives a persistent track ID so that identity, gestures,
activities, expressions, and review state can be maintained independently over
time.

### Pose Estimation

YOLO Pose estimates 17 body keypoints, including:

* shoulders
* elbows
* wrists
* hips
* knees
* ankles

The pose landmarks are used for skeleton visualization, wave detection, and
pose-based activity recognition.

### Identity Recognition

Optional identity recognition uses OpenCV DNN models:

* YuNet for face detection
* SFace for face recognition

Identity confirmation requires agreement across multiple frames. Low-quality
faces are rejected, and confirmed identities expire unless they are
revalidated.

The application continues anonymously when no enrolled identities are
available.

<img width="1118" height="624" alt="623024201-546550a0-efeb-4ee6-baf0-524638fbcb93" src="https://github.com/user-attachments/assets/27bed33e-6ba6-478d-855f-aaa41191843d" />

### Facial-Expression Context

MediaPipe is used to locate visible faces within tracked person regions.
EmotiEffLib then estimates facial-expression context.

A rolling smoothing window reduces label flicker. Lower-confidence estimates
can remain visible with a `?` indicator to show that the classification is
tentative.

Expression estimates are informational and do not directly change the
behavior-based review level.

### Repeated-Wave Detection

The system detects completed raised right-hand wave gestures using temporal
pose information.

Wave events are counted within a rolling time window. The first two completed
waves are treated as ordinary activity, while additional repeated waves
increase the review level.

### Review Levels

The application uses four behavior-based review states:

* `CLEAR`: green
* `MONITOR`: yellow
* `REVIEW`: orange
* `HIGH`: red

The review state is based on repeated temporal gesture activity. Identity,
facial expression, and activity classifications remain separate contextual
signals.

<img width="800" height="533" alt="state_isolation_sequence_annotated-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/24ff5e5b-df0e-4e5f-a347-b9d4f4f8dbff" />

### Independent Multi-Person State

Each tracked person has an independent:

* identity state
* wave history
* expression history
* activity sequence
* review level
* event history

Histories are isolated by tracker ID. ID switches during crossings or occlusion
can still associate history with the wrong person.

## Human Activity Recognition

The project includes an activity-recognition benchmark for:

* walking
* running
* standing
* sitting

Three approaches were evaluated using the same source-video partitions.

### MLP Pose Model

The MLP uses sequences of normalized YOLO Pose landmarks.

Each sample contains 16 frames of:

* 17 keypoint x-coordinates
* 17 keypoint y-coordinates
* 17 keypoint visibility values

The saved MLP checkpoint can also be enabled in the main application to provide
informational per-person activity labels.

### MobileNetV2

The image-based approach uses cached features extracted from a frozen
ImageNet-pretrained MobileNetV2 backbone.

A four-class linear classification head is trained using the cached features.

### S3D

The advanced video approach uses cached features extracted from a frozen
Kinetics-400-pretrained S3D model.

A four-class linear classification head is trained on the resulting temporal
video features.

## Activity Recognition Results

The models were evaluated using 120 held-out HMDB51 test videos.
These are historical benchmark results, not a rerun against the current
dependency versions and source-time preprocessing. Reproduce the benchmark
before using these figures to assess the current release.

| Model       | Accuracy | Macro F1 | Head Training Time | CPU Inference Latency |
| ----------- | -------: | -------: | -----------------: | --------------------: |
| MLP         |   46.67% |   47.00% |            0.219 s |              0.262 ms |
| MobileNetV2 |   47.50% |   46.02% |            0.111 s |              80.73 ms |
| S3D         |   42.50% |   41.35% |            0.065 s |             293.31 ms |

MobileNetV2 achieved the highest accuracy, while the MLP produced the highest
macro F1 and substantially lower inference latency.

The MLP was selected for optional real-time integration because it reuses the
pose landmarks already generated by the tracking pipeline and does not require
an additional image or video backbone during inference.

<img width="1600" height="640" alt="10_training_curves" src="https://github.com/user-attachments/assets/75561f0f-dc47-4e7e-a4bb-dda5674cdb05" /><img width="864" height="736" alt="13_s3d_confusion_matrix" src="https://github.com/user-attachments/assets/1a26418a-3605-4d0c-b60d-9f2f4974ee45" />
<img width="864" height="736" alt="12_mobilenetv2_confusion_matrix" src="https://github.com/user-attachments/assets/43588d4e-4cb0-4950-ab2d-3c2f789de65c" />
<img width="864" height="736" alt="11_mlp_confusion_matrix" src="https://github.com/user-attachments/assets/aad573e2-0b83-4f72-8d9b-10161263ca24" />


Additional measured results and interpretation are available in:

```text
docs/activity_recognition_results.md
```

## System Architecture

The processing pipeline is divided into independent components:

1. Video frame acquisition
2. YOLO Pose person detection
3. ByteTrack identity assignment
4. Pose landmark extraction
5. Gesture-state updates
6. Optional face and identity analysis
7. Facial-expression estimation and smoothing
8. Optional MLP activity inference
9. Per-person temporal state updates
10. Review-level calculation
11. Annotation rendering
12. Structured event logging

The activity model reuses YOLO Pose results already generated by the main
pipeline. It does not run an additional pose model.

## Technology Stack

### Computer Vision

* OpenCV
* Ultralytics YOLOv8 Pose
* ByteTrack
* MediaPipe
* YuNet
* SFace
* EmotiEffLib

### Machine Learning

* PyTorch
* Torchvision
* NumPy
* MobileNetV2
* S3D
* multilayer perceptron

### Development and Validation

* Python 3.11 and 3.12
* GitHub Actions
* Python `unittest`
* JSONL event logging
* manifest-based validation
* checksum-verified model provisioning

## Repository Structure

```text
computer-vision-behavioral-analysis/
├── .github/
│   └── workflows/
├── data/
├── docs/
│   ├── configuration.md
│   └── data-governance.md
├── input/
├── output/
├── scripts/
├── src/
│   ├── activity_recognition/
│   └── monitoring_assets/
├── tests/
├── validation/
├── pyproject.toml
├── requirements.txt
├── requirements-tested.txt
├── requirements-activity.txt
└── README.md
```

### Main Directories

* `src/` contains the main application and computer vision components.
* `src/activity_recognition/` contains the activity dataset, preprocessing,
  models, training, evaluation, metrics, and optional live MLP integration.
* `scripts/` contains activity preparation, training, evaluation, and isolated
  wheel-installation verification entry points.
* `tests/` contains the automated unit and integration tests.
* `validation/` contains the scenario-based validation framework.
* `docs/` contains technical documentation and experiment results.
* `input/` stores local photos or videos selected for processing.
* `output/` stores generated media, logs, events, and benchmark results.
* `data/` stores local model files, datasets, feature caches, and checkpoints.

Generated files, downloaded datasets, checkpoints, enrollments, logs, and
outputs are excluded from version control.

## Installation

Python 3.11 or 3.12 is required by the package metadata.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install the core dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare and verify the required model files:

```powershell
.\.venv\Scripts\python.exe src\main.py --doctor --prepare-models
```

On macOS or Linux, replace:

```text
.\.venv\Scripts\python.exe
```

with:

```text
.venv/bin/python
```

### Exact Tested Environment

`requirements-tested.txt` is a historical July 2026 Windows snapshot of
top-level package versions, with the current Torch security floor retained.
It is not a complete dependency lockfile or the latest audited environment.
Use `requirements.txt` for normal installation.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
```

### Optional Activity Dependencies

Install the activity-recognition dependencies when training, evaluating, or
using the MLP activity checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-activity.txt
```

## Running the Application

Commands below using plain `python` assume the virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux use `source .venv/bin/activate`. Alternatively, use the explicit
virtual-environment Python path shown in Installation for every command.

Open the application menu:

```powershell
python src/main.py
```

The menu provides options for:

* enrolling identities
* starting real-time processing
* deleting existing enrollments
* processing recorded photos and videos

To start processing directly:

```powershell
python src/main.py --detect --source 0
```

Camera source `0` generally represents the built-in camera. Other camera
indices, video paths, and supported stream sources can also be provided.

Press `q` to close the processing window.

## Processing Recorded Media

Place a supported image or video in the `input` directory:

```powershell
python src/main.py --process-media
```

Annotated files are written to the `output` directory.
Video exports do not preserve the source audio. Failed direct CLI processing
returns a nonzero exit status; the interactive menu retains best-effort export.

Video output names retain the source extension, so `clip.mov` becomes
`clip.mov_annotated.mp4` and `clip.avi` becomes `clip.avi_annotated.mp4`.
This prevents different input formats with the same base name from overwriting
each other. Reprocessing the same input replaces its previous output atomically.

Recorded sources use video timestamps for gesture windows and activity sampling,
with a frame-rate fallback when the decoder does not provide usable timestamps.
Live cameras and streams use monotonic elapsed time.

To process one specific file:

```powershell
python src/main.py --process-media --input "input\sample.mp4"
```

Supported image formats include:

* `.bmp`
* `.jpeg`
* `.jpg`
* `.png`
* `.webp`

Supported video formats include:

* `.avi`
* `.m4v`
* `.mkv`
* `.mov`
* `.mp4`

## Enabling Live Activity Recognition

Newly prepared activity caches and checkpoints use a shared source-time sampling
interval. See [activity preprocessing and checkpoint compatibility](docs/activity_preprocessing.md)
for rebuild commands and legacy checkpoint behavior.

Prepare the dataset/cache using the Activity Benchmark instructions below,
then train the activity models:

```powershell
.\.venv\Scripts\python.exe scripts\train_activity_models.py
```

The default MLP checkpoint is generated at:

```text
data/activity_models/mlp.pt
```

Enable the saved MLP checkpoint:

```powershell
.\.venv\Scripts\python.exe src\main.py --detect --source 0 `
  --activity-model mlp `
  --activity-checkpoint "data\activity_models\mlp.pt"
```

The activity recognizer:

* waits for 16 valid pose frames
* performs inference every five frames by default
* smooths the five most recent probability vectors
* displays low-confidence predictions as `Unknown`
* stores separate pose and prediction histories for each tracked person

## Activity Benchmark

Prepare the HMDB51 data:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_activity_data.py `
  --dataset-root "data\hmdb51\hmdb51_org" `
  --annotations-root "data\hmdb51\splits"
```

Train and evaluate all three models:

```powershell
.\.venv\Scripts\python.exe scripts\train_activity_models.py
```

Evaluate existing checkpoints without retraining:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_activity_models.py
```

Generated results are written to:

```text
output/activity_results/
```

The generated files include:

* `metrics.json`
* `model_comparison.csv`
* `predictions.csv`
* `mlp_confusion_matrix.png`
* `cnn_confusion_matrix.png`
* `advanced_confusion_matrix.png`
* `training_curves.png`
* `run_manifest.json`

## Validation

Copy the example validation manifest:

```powershell
Copy-Item validation\manifest.example.json validation\manifest.json
```

Add consented validation media under:

```text
validation/media/
```

Run the validation framework:

```powershell
.\.venv\Scripts\python.exe src\main.py --validate `
  --manifest validation\manifest.json `
  --report validation\reports\latest.json
```

The validation runner measures:

* number of people detected
* confirmed identities
* review-state counts
* processing performance
* scenario acceptance criteria

## Testing

Install both the core and optional activity dependencies before running the
complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-activity.txt
```

Check dependency consistency:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Compile the source and tests:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite covers temporal state, model loading, preprocessing, media export,
and event reporting. Run it in your installed environment to verify compatibility.

CI also checks lint, formatting, types, distribution builds and installation
outside the checkout. With both dependency sets installed, run these locally:

```powershell
.\.venv\Scripts\python.exe -m pip install ruff==0.16.7 pyright==1.1.414 build
.\.venv\Scripts\python.exe -m ruff check --config pyproject.toml src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check --config pyproject.toml src scripts tests
.\.venv\Scripts\python.exe -m pyright src scripts
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe scripts/verify_wheel.py
```

Install the repository as an editable Python project:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Model Provisioning

The application automatically downloads several required models when they are
not present locally.

Downloads use:

* pinned upstream revisions
* expected SHA-256 checksums
* temporary download files
* atomic file replacement
* configurable retry behavior

Prepare all downloadable models:

```powershell
python src/main.py --doctor --prepare-models
```

Verify operation without allowing new downloads:

```powershell
python src/main.py --doctor --no-model-downloads
```

The MediaPipe face detector model at
`src/monitoring_assets/blaze_face_short_range.tflite` is included in source
and wheel distributions and loaded as a read-only package asset.

### Standalone wheel installation

With Python 3.11 or 3.12, build and install the wheel:

```powershell
python -m pip install build
python -m build
python -m pip install dist/integrated_monitoring_poc-0.2.1-py3-none-any.whl
integrated-monitoring --doctor --prepare-models
integrated-monitoring --doctor --no-model-downloads
integrated-monitoring --detect
```

The wheel can be copied to another machine and installed there without the
repository. Installation still requires the Python dependencies; first model
preparation requires network access. In a wheel installation, downloaded
models, enrollments, logs and menu media folders default to
`~/.integrated-monitoring-poc`. Set `MONITOR_HOME` before launch to use another
writable directory. Source runs retain the checkout root by default.
Explicit CLI media paths remain relative to the working directory.
`python -m main` also works after installation.

To verify the wheel outside the checkout with the currently installed
dependencies, run `python scripts/verify_wheel.py`. This creates a temporary
installation, checks the launcher and paths, and loads the bundled face model.

Other downloaded model files are excluded from version control.

## Design Considerations

### Temporal Processing

Many computer vision systems evaluate each frame independently. This project
uses temporal state to support:

* repeated-gesture detection
* identity confirmation
* identity expiration
* expression smoothing
* pose-sequence activity recognition
* review-level progression
* stale-track cleanup

### Shared Pose Inference

The activity-recognition system does not run YOLO Pose a second time.

The existing pose results are reused for:

* skeleton rendering
* wave detection
* activity feature generation

This reduces redundant computation.

### Independent Track State

State is keyed by the persistent tracking ID. This isolates histories while
tracker IDs remain correct; tracker ID switches during crossings or occlusion
can still associate history with the wrong person and require validation.

### Local Model Readiness and Telemetry

After model preparation, verify local model readiness without downloading:

```powershell
python src/main.py --doctor --no-model-downloads
```

This flag does not enforce network isolation. Ultralytics analytics and ONNX
Runtime telemetry are disabled before model initialization. MediaPipe's
published binary still has no supported telemetry opt-out; controlling its
outbound traffic or using a telemetry-free source build remains unresolved.
See [runtime configuration](docs/configuration.md) for details and upstream references.

Datasets, generated checkpoints, feature caches, reports, and output media are
kept outside Git version control.

## Data Access and Retention

Run the application and maintenance commands under the intended operating-system
account. New managed data directories are private. To restrict existing output,
enrollment, event and log directories, stop monitoring and run:

```powershell
python src/main.py --secure-data
```

Windows grants access to the current account and SYSTEM. POSIX migration uses
0700 directories and 0600 files. These controls do not provide application
login, encryption, or separation between people sharing the same account.

| Managed data | Default retention | Environment override |
| --- | --- | --- |
| Annotated media | 30 days | `MONITOR_RECORDING_RETENTION_DAYS` |
| Enrollment images | 365 days | `MONITOR_IDENTITY_RETENTION_DAYS` |
| Events and application logs | 30 days | `MONITOR_LOG_RETENTION_DAYS` |

Retention uses file modification time and requires explicit maintenance:

```powershell
python src/main.py --retention
# After reviewing the preview, with monitoring stopped:
python src/main.py --retention --apply-retention
```

No background deletion is enabled. Original inputs, backups and custom export
directories outside the managed output root are excluded. See the
[data access and retention policy](docs/data-governance.md) for scope and safeguards.

## Limitations

* Activity recognition was evaluated using a limited four-class subset of
  HMDB51.
* The benchmark contained 120 held-out test videos.
* Performance may change under different lighting, camera angles, clothing,
  movement speeds, occlusion, and environmental conditions.
* Facial-expression estimates do not establish a person's intent or emotional
  state.
* Identity recognition depends on enrollment quality and visible facial
  features.
* Review levels are demonstration-oriented behavioral indicators and are not
  security conclusions.
* The benchmark models were not trained end to end.
* MobileNetV2 and S3D use frozen pretrained feature extractors.
* Real-world deployment would require broader validation, privacy review,
  fairness analysis, security controls, and legal assessment.

## Responsible Use

This project is intended for academic evaluation, technical experimentation,
and portfolio presentation.

It does not determine whether a person is suspicious, dangerous, or engaged in
wrongdoing.

Pose, identity, expression, and activity predictions can be incorrect or
affected by:

* lighting
* camera placement
* occlusion
* image quality
* appearance
* disability
* environmental conditions
* model and dataset limitations

Any real-world application would require:

* informed consent
* appropriate data governance
* human oversight
* access controls
* bias and fairness evaluation
* privacy protection
* compliance with applicable laws and organizational policies
