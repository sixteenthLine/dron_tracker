# Project Status

Detailed handoff for `falcon-vision-reconstructed`. The original August validation
commands below are retained as evidence; current commands use the licensed
`assets/aerial/park.jpg` and `assets/aerial/terrain.jpg` samples and are listed in
[RUNBOOK.md](RUNBOOK.md). Execute commands from the repository root:

The original Level B status sections below are a dated handoff, not the current
training status. Offline training was subsequently added on October 22; see
[TRAINING.md](TRAINING.md), [training evidence](validation/TRAINING_VALIDATION.md),
and the current [root status](../PROJECT_STATUS.md). Colab remains unexecuted.

```bash
cd falcon-vision-reconstructed
```

## 1. Current Project State

Status: implemented and tested for a safe macOS CPU portfolio demo.

The project is a Python package for generic object detection, deterministic synthetic video processing, simple multi-object tracking, visualization overlays, structured logs, optional ROS2 perception output, tests, and documentation. It does not implement training, drone control, navigation, autonomous following, targeting, trigger logic, ONNX export, Docker, or deployment.

## 2. Historical Project

Historical evidence showed two separate prototype branches:

- FalconVision detector/data-loading work: COCO-style dataset loader and Faster R-CNN notebooks.
- Separate OpenCV/GOTURN tracking and drone-adjacent ROS material.

No evidence showed completed detector training, historical metrics, a recovered detector checkpoint, or historical detector/tracker/ROS integration.

## 3. Recovered Directly

Status: recovered as evidence only.

- COCO-style dataset-loading concept.
- Notebook evidence that Faster R-CNN training began, then stopped.
- Separate OpenCV/GOTURN tracking artifacts.
- A manual ROS teleoperation artifact, excluded from this package.

No historical source files were copied into the reconstructed package as implementation.

## 4. Rewritten

Status: implemented and tested.

- `CocoDetectionDataset`: rewritten from the historical loader idea with no hard-coded paths, correct `xywh` to `xyxy`, configurable labels, empty-box handling, type hints, and validation errors.
- Detector factory/CLI wrapper around modern public torchvision pretrained Faster R-CNN.
- Visualization helpers for boxes and labels.

## 5. Newly Added

Implementation is present; tested scope is defined in the current root status
table and dated validation records. Availability alone is not validation.

- Package structure under `src/falcon_vision_reconstructed`.
- CLI command group `falcon`.
- Frame interfaces, models, config, sources, processor, video writer, logging.
- `MockDetector`, `MockTracker`, `IoUTracker`, `NoOpTracker`.
- Legacy GOTURN compatibility boundary: stub only.
- ROS2 camera bridge: implemented but ROS2 runtime not tested because ROS2 was unavailable.
- Synthetic demo video utilities and safe public image asset.
- Tests and Level A/Level B validation docs.

## 6. Architecture And Data Flow

Status: image CLI, synthetic/video processing with mock detector, and IoU
association were tested. The standalone image source, webcam, NoOpTracker, and
torchvision video path have not been validated end to end.

```text
FrameSource
  -> Detector
  -> Tracker
  -> Visualization overlay
  -> OutputVideoWriter
  -> structured JSON logs
```

Available sources: image file, video file, optional webcam, deterministic synthetic source.

Available detectors: torchvision detector, mock detector.

Available trackers: IoU tracker, no-op tracker, mock tracker, legacy GOTURN stub.

Optional ROS2 flow:

```text
/camera/image_raw
  -> ROS image conversion
  -> detector/tracker
  -> /falcon/detections
  -> /falcon/tracks
  -> /falcon/debug_image
  -> /falcon/status
```

ROS2 publishes perception/debug/status only. It does not publish control topics.

## 7. Available CLI Commands

Run from `falcon-vision-reconstructed`.

| command | status |
|---|---|
| `.venv-level-b/bin/falcon infer-image ...` | implemented and tested |
| `.venv-level-b/bin/falcon infer-video ...` | implemented and tested with synthetic video file |
| `.venv-level-b/bin/falcon run-pipeline --config ...` | implemented and tested with synthetic config |
| `.venv-level-b/bin/falcon create-demo-video ...` | implemented and tested |
| `.venv-level-b/bin/falcon ros-node --check-only` | implemented and tested |
| `.venv-level-b/bin/falcon ros-node` | implemented but untested; blocked by missing ROS2 runtime |
| `.venv-level-b/bin/falcon validate` | implemented and tested |
| `.venv-level-b/bin/falcon-vision-demo --input ...` | implemented for Level A compatibility; tested earlier in Level A, not retested in Level B |

## 8. Installation Commands

Run from `falcon-vision-reconstructed`.

```bash
python3.11 -m venv .venv-level-b
.venv-level-b/bin/python -m pip install --upgrade pip
.venv-level-b/bin/python -m pip install -r requirements.txt
.venv-level-b/bin/python -m pip install -e .
```

Status: executed successfully.

## 9. Test, Demo, And Validation Commands

### Unit Tests

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/python -m pytest tests/test_coco_dataset.py tests/test_tracking.py tests/test_ros_bridge.py
```

Status: available but not executed as this exact split command. Covered by the full successful `pytest` run.

### Integration Tests

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/python -m pytest tests/test_pipeline.py
```

Status: available but not executed as this exact split command. Covered by the full successful `pytest` run.

### Full Test Suite

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/python -m pytest
```

Status: executed successfully, `21 passed`.

### Image Inference

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon infer-image \
  --input assets/sample_civilian_dog.jpg \
  --output outputs/sample_civilian_dog_detected_level_b.jpg \
  --threshold 0.50
```

Status: executed successfully. Output saved to `outputs/sample_civilian_dog_detected_level_b.jpg`.

### Video Inference

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon infer-video \
  --input outputs/synthetic_civilian_input.mp4 \
  --output outputs/synthetic_civilian_infer_video.mp4 \
  --detector mock \
  --tracker iou \
  --threshold 0.50 \
  --max-frames 8 \
  --fps 12
```

Status: executed successfully after generating the input video. Output saved to `outputs/synthetic_civilian_infer_video.mp4`.

### Tracking Demo

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon run-pipeline --config configs/synthetic_video_demo.toml
```

Status: executed successfully. Uses built-in synthetic source, mock detector, and IoU tracker. Output saved to `outputs/synthetic_civilian_tracked.mp4`.

### Synthetic Input Video Creation

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon create-demo-video \
  --output outputs/synthetic_civilian_input.mp4 \
  --frames 24 \
  --fps 12
```

Status: executed successfully. Output saved to `outputs/synthetic_civilian_input.mp4`.

### ROS2 Or Mocked ROS Demo

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon ros-node --check-only
```

Status: executed successfully. Result showed `ros2_available: false`.

```bash
.venv-level-b/bin/falcon ros-node
```

Status: not executed successfully; blocked by missing ROS2 runtime in the validation environment.

### Validation

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon validate
```

Status: executed successfully. Result showed ROS2 unavailable and listed safe topics.

## 10. Required Input Files

| demo | required input |
|---|---|
| image inference | `assets/sample_civilian_dog.jpg` |
| create demo video | none beyond package install |
| video inference | `outputs/synthetic_civilian_input.mp4`, generated by `create-demo-video` |
| tracking demo | `configs/synthetic_video_demo.toml`; built-in synthetic source |
| ROS2 check-only | none beyond package install |
| ROS2 node | ROS2 Python runtime and camera topic `/camera/image_raw` |
| full validation | installed environment plus files in this project |

## 11. Generated Output Paths

Status: generated and confirmed in Level B validation.

- `outputs/sample_civilian_dog_detected_level_b.jpg`
- `outputs/synthetic_civilian_input.mp4`
- `outputs/synthetic_civilian_tracked.mp4`
- `outputs/synthetic_civilian_infer_video.mp4`

## 12. Commands Actually Executed Successfully

Run from `falcon-vision-reconstructed`; all succeeded:

```bash
python3.11 -m venv .venv-level-b
.venv-level-b/bin/python -m pip install --upgrade pip
.venv-level-b/bin/python -m pip install -r requirements.txt
.venv-level-b/bin/python -m pip install -e .
.venv-level-b/bin/python -m pytest
.venv-level-b/bin/falcon infer-image --input assets/sample_civilian_dog.jpg --output outputs/sample_civilian_dog_detected_level_b.jpg --threshold 0.50
.venv-level-b/bin/falcon create-demo-video --output outputs/synthetic_civilian_input.mp4 --frames 24 --fps 12
.venv-level-b/bin/falcon run-pipeline --config configs/synthetic_video_demo.toml
.venv-level-b/bin/falcon infer-video --input outputs/synthetic_civilian_input.mp4 --output outputs/synthetic_civilian_infer_video.mp4 --detector mock --tracker iou --threshold 0.50 --max-frames 8 --fps 12
.venv-level-b/bin/falcon ros-node --check-only
.venv-level-b/bin/falcon validate
```

## 13. Commands Not Executed

Run from `falcon-vision-reconstructed`.

```bash
.venv-level-b/bin/falcon ros-node
.venv-level-b/bin/falcon infer-video --input <video> --output <video> --detector torchvision --tracker iou
.venv-level-b/bin/falcon infer-video --input <video> --output <video> --tracker legacy_goturn
.venv-level-b/bin/falcon infer-video --input <webcam-or-video> --tracker mock
.venv-level-b/bin/falcon-vision-demo --input assets/sample_civilian_dog.jpg --output outputs/sample_civilian_dog_detected.jpg --threshold 0.50
```

The legacy `falcon-vision-demo` path was validated during Level A, but not re-executed during the final Level B validation.

## 14. Known Errors And Limitations

- A local Heridal export with annotations is now present, but its equivalence to
  the historical training data is unverified. It is ignored by Git and unused by
  demos; see [DATASET_STATUS.md](DATASET_STATUS.md).
- No historical checkpoint is present.
- No historical metrics are present or claimed.
- No training is implemented.
- Torchvision image inference may download public model weights on first use.
- Synthetic video validates integration, not model quality.
- IoU tracking is a simple generic tracker, not a production MOT benchmark.
- ROS2 runtime was unavailable during validation.
- GOTURN is unavailable by default.
- Webcam source is implemented but untested and requires optional OpenCV.
- No CUDA is required or tested; validation used CPU on macOS.

## 15. Optional Dependencies

Status:

- `legacy-goturn`: optional extra declares `opencv-contrib-python-headless==4.12.0.88`; not installed or tested.
- `ros`: optional extra is empty because ROS2 is normally installed outside pip; ROS2 runtime not available in validation.
- Webcam input depends on local OpenCV availability; not installed or tested.

Core dependencies are pinned in `requirements.txt`.

## 16. GOTURN Status

Status: stub only. Installing OpenCV or supplying model paths does not enable it.

GOTURN does not currently work in the reconstructed package. The `legacy_goturn` tracker backend intentionally raises `LegacyGoturnUnavailable`. No GOTURN model files are redistributed, and no historical files are modified.

## 17. Modern Tracker Status

Status: implemented and tested.

`IoUTracker` works for the synthetic demo and tests. It provides generic track IDs, bounding-box updates, configurable IoU threshold, lifecycle handling, and no identity recognition or control output.

## 18. ROS2 Integration Status

Status: implemented but runtime untested; mocked helper tests passed.

`ros-node --check-only` and `falcon validate` were tested successfully and reported `ros2_available: false`. The actual ROS2 node was not run because ROS2 was not installed in the validation environment.

## 19. macOS Without CUDA

Status: implemented and tested.

The project was validated on macOS arm64 with Python 3.11.11 using CPU/default device settings. CUDA is not required for the tested commands.

## 20. Simplest Recommended First Command

Run from `falcon-vision-reconstructed` after installation:

```bash
.venv-level-b/bin/falcon run-pipeline --config configs/synthetic_video_demo.toml
```

This is the simplest Level B demo because it needs no external media, no ROS2, no GOTURN, no CUDA, and no model download.
