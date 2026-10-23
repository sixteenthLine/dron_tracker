# Level B Validation

This is the August 2, 2026 validation record, not a claim about every current
backend. Its former image sample is excluded from publication. See
[the current handoff](../PROJECT_STATUS.md) for the licensed replacement sample.

Validation was run from the `falcon-vision-reconstructed` project root on August 2, 2026.

## Environment

- Python: 3.11.11
- Platform: macOS arm64
- Fresh environment: `.venv-level-b`
- Core package: editable install of `falcon-vision-reconstructed==0.1.0`
- Key runtime dependencies:
  - `torch==2.5.1`
  - `torchvision==0.20.1`
  - `Pillow==11.0.0`
  - `imageio==2.37.0`
  - `imageio-ffmpeg==0.6.0`
  - `pytest==8.3.4`

## Commands And Results

```bash
python3.11 -m venv .venv-level-b
```

Result: exit code 0.

```bash
.venv-level-b/bin/python -m pip install --upgrade pip
```

Result: exit code 0; `pip` upgraded to 26.2.

```bash
.venv-level-b/bin/python -m pip install -r requirements.txt
```

Result: exit code 0; pinned dependencies installed successfully.

```bash
.venv-level-b/bin/python -m pip install -e .
```

Result: exit code 0; editable package installed successfully.

```bash
.venv-level-b/bin/python -m pytest
```

Result:

```text
collected 21 items
tests/test_coco_dataset.py .....                                         [ 23%]
tests/test_pipeline.py .....                                             [ 47%]
tests/test_ros_bridge.py ....                                            [ 66%]
tests/test_tracking.py .......                                           [100%]
21 passed in 1.81s
```

```bash
.venv-level-b/bin/falcon infer-image \
  --input assets/sample_civilian_dog.jpg \
  --output outputs/sample_civilian_dog_detected_level_b.jpg \
  --threshold 0.50
```

Result:

```text
Saved outputs/sample_civilian_dog_detected_level_b.jpg with 1 detections at threshold 0.50 using torchvision Faster R-CNN MobileNet V3 Large 320 FPN, COCO pretrained weights.
```

```bash
.venv-level-b/bin/falcon create-demo-video \
  --output outputs/synthetic_civilian_input.mp4 \
  --frames 24 \
  --fps 12
```

Result:

```text
Created synthetic demo video: outputs/synthetic_civilian_input.mp4
```

```bash
.venv-level-b/bin/falcon run-pipeline --config configs/synthetic_video_demo.toml
```

Result: exit code 0. Structured per-frame logs showed 24 processed synthetic frames, one mock detection per frame, one active IoU track per frame, and latency measurements.

Final summary:

```json
{"average_latency_ms": 0.013, "frames_processed": 24, "frames_seen": 24, "output_path": "outputs/synthetic_civilian_tracked.mp4"}
```

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

Result: exit code 0. Structured logs showed 8 processed video-file frames, one mock detection per frame, one active IoU track per frame, and latency measurements.

Final summary:

```json
{"average_latency_ms": 0.033, "frames_processed": 8, "frames_seen": 8, "output_path": "outputs/synthetic_civilian_infer_video.mp4"}
```

```bash
.venv-level-b/bin/falcon ros-node --check-only
```

Result:

```json
{"ros2_available": false, "topics": {"camera": "/camera/image_raw", "debug_image": "/falcon/debug_image", "detections": "/falcon/detections", "status": "/falcon/status", "tracks": "/falcon/tracks"}}
```

ROS2 was not installed in this fresh environment. The ROS2 node was therefore not started. Mocked/non-runtime ROS helper tests passed through `tests/test_ros_bridge.py`.

```bash
.venv-level-b/bin/falcon validate
```

Result:

```json
{"ros2_available": false, "topics": {"camera": "/camera/image_raw", "debug_image": "/falcon/debug_image", "detections": "/falcon/detections", "status": "/falcon/status", "tracks": "/falcon/tracks"}}
```

## Generated Outputs

```text
outputs/sample_civilian_dog_detected_level_b.jpg: exists=True bytes=270488
outputs/synthetic_civilian_input.mp4: exists=True bytes=3532
outputs/synthetic_civilian_tracked.mp4: exists=True bytes=4520
outputs/synthetic_civilian_infer_video.mp4: exists=True bytes=3289
```

## Passed Test Areas

- COCO dataset utility:
  - one image with one box;
  - one image with multiple boxes;
  - one image with no boxes;
  - missing image;
  - malformed annotation.
- Tracker behavior:
  - IoU calculation;
  - stable IDs;
  - multiple objects;
  - lifecycle expiration;
  - small-box filtering;
  - explicit mock tracker behavior;
  - legacy GOTURN unavailable by default.
- Pipeline behavior:
  - synthetic frame source;
  - mock detector;
  - IoU tracker;
  - frame skipping;
  - video read/write smoke test;
  - typed TOML config.
- ROS bridge helpers:
  - safe topic contract;
  - JSON detection/track/status payloads;
  - ROS image conversion helper without active ROS runtime;
  - ROS availability check.

## Known Limitations

- No training was started or added.
- No historical model-training completion is claimed.
- No historical performance metrics are claimed.
- The image detector uses modern public torchvision pretrained weights, not historical weights.
- The synthetic video demo validates integration and reproducibility, not model accuracy.
- The IoU tracker is a simple generic tracker for safe demos, not a production MOT benchmark.
- ROS2 runtime was unavailable in the validation environment, so only availability and mocked helper tests were run.

## Unsupported Optional Components

- Legacy GOTURN is not enabled by default and no GOTURN model files are redistributed.
- ROS2 is optional and absent from the fresh validation environment.
- Webcam input is optional and requires local OpenCV support.
- Real-drone integration, simulated drone control, navigation, autonomous following, targeting, trigger logic, and operational deployment are not implemented.
