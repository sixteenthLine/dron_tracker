# Runbook

All commands below run from the **repository root** (`falcon-vision-reconstructed`).
Use Python 3.11. Python 3.12 is declared compatible but not locally validated.
The public repository/distribution is now `aerial-object-detection`; a fresh
clone normally has that folder name. The local folder and legacy imports remain
unchanged for compatibility. New CLI names are `aerial-detect` and `aerial-train`.

## Installation

```bash
python3.11 -m venv .venv-level-b
.venv-level-b/bin/python -m pip install --upgrade pip
.venv-level-b/bin/python -m pip install -r requirements.txt
.venv-level-b/bin/python -m pip install -e .
.venv-level-b/bin/python -m pip check
```

The original fresh-environment install passed on macOS arm64. Current checks
and any package-refresh results are recorded in [validation](validation/GITHUB_PREPARATION.md).
Linux CPU dependency installation is configured in CI but awaits a hosted run.

## Commands, Inputs, Outputs, And Status

| CLI command | Required input | Output | Verified scope |
|---|---|---|---|
| `infer-image` | Image file, output path; first run needs weights/cache | Output image | CPU inference on two aerial frames; 0 predictions at 0.50 |
| `create-demo-video` | No media | Synthetic MP4 | Tested |
| `infer-video` | Video file and output path | Annotated MP4 and logs | Tested with mock detector and IoU |
| `run-pipeline` | TOML config; media only for file sources | MP4 and logs | Tested with built-in synthetic source |
| `ros-node --check-only` | No media or ROS graph | Availability JSON | Tested; ROS2 unavailable |
| `ros-node` | ROS2 runtime and `/camera/image_raw` | Perception topics | Implemented, runtime untested |
| `validate` | Installed package | Availability/topic JSON | Tested; does not run tests or inference |
| `falcon-vision-demo --input ...` | Image and output path | Annotated image | Backward-compatible CLI alias |

`aerial-detect` and the older `falcon` / `falcon-vision-demo` aliases call the
same CLI. `aerial-train` and `falcon-train` call the training CLI. Image options:
`--input`, `--output`, `--threshold`, `--max-detections`, `--device`.
Video options also include `--frame-skip`, `--max-frames`, `--fps`, `--log-level`,
`--detector {mock,torchvision}` and `--tracker {iou,none,mock,legacy_goturn}`.
The video CLI defaults to **mock** detections. `legacy_goturn` always raises
an unavailable error. Use `infer-video --help` for the video parser's help;
the bare top-level `--help` currently routes through the legacy image parser.

## Tests

Unit tests and ROS helpers (16 tests):

```bash
.venv-level-b/bin/python -m pytest -q tests/test_coco_dataset.py tests/test_tracking.py tests/test_ros_bridge.py
```

Pipeline test file (5 tests, including config cases and video I/O integration):

```bash
.venv-level-b/bin/python -m pytest -q tests/test_pipeline.py
```

Full suite:

```bash
.venv-level-b/bin/python -m pytest -q
```

Tests create temporary synthetic fixtures; no local dataset or model weights
are required. ROS helper tests do not start a node or prove ROS topic delivery.

## Image Inference

```bash
.venv-level-b/bin/aerial-detect infer-image --input assets/aerial/park.jpg --output outputs/aerial_park_detected.jpg --threshold 0.50
.venv-level-b/bin/aerial-detect infer-image --input assets/aerial/terrain.jpg --output outputs/aerial_terrain_detected.jpg --threshold 0.50
```

Both full-frame inputs are included with [attribution](../assets/SAMPLE_SOURCE.md).
The unchanged generic model returned zero predictions on each at 0.50; successful
file generation does not establish aerial detection accuracy. First run may
download weights. For the older alias syntax, the equivalent command is:

```bash
.venv-level-b/bin/falcon-vision-demo --input assets/aerial/park.jpg --output outputs/aerial_park_detected_alias.jpg --threshold 0.50
```

## Synthetic Video And Tracking

The first command requires no input media; it creates the file needed by the third.
The middle command uses built-in synthetic frames and does not read the MP4.

```bash
.venv-level-b/bin/aerial-detect create-demo-video --output outputs/synthetic_civilian_input.mp4 --frames 24 --fps 12
.venv-level-b/bin/aerial-detect run-pipeline --config configs/synthetic_video_demo.toml
.venv-level-b/bin/aerial-detect infer-video --input outputs/synthetic_civilian_input.mp4 --output outputs/synthetic_civilian_infer_video.mp4 --detector mock --tracker iou --max-frames 8 --fps 12
```

Outputs: `outputs/synthetic_civilian_input.mp4` (24 frames),
`outputs/synthetic_civilian_tracked.mp4` (24 frames), and
`outputs/synthetic_civilian_infer_video.mp4` (8 frames). The tracker performs
actual IoU association; detections in these videos are synthetic mock results.

## ROS2 Helpers And Validation

```bash
.venv-level-b/bin/aerial-detect ros-node --check-only
.venv-level-b/bin/python -m pytest -q tests/test_ros_bridge.py
.venv-level-b/bin/aerial-detect validate
```

These commands are available without ROS2. For an actual ROS2 environment,
source its installation in your shell, ensure its Python modules are available
to the interpreter, and provide safe images on `/camera/image_raw` before running:

```bash
.venv-level-b/bin/aerial-detect ros-node
```

That startup/topic-delivery path is **untested and blocked locally by missing
ROS2 dependencies**. The node currently uses a fixed mock detector. It publishes
only `/falcon/detections`, `/falcon/tracks`, `/falcon/debug_image`, `/falcon/status`.
There is no bundled camera publisher, ROS bag, or drone integration.

## Documentation Previews

From the root, the new main README previews can be rebuilt without model weights
or the full dataset:

```bash
.venv-level-b/bin/python scripts/render_reference_previews.py
```

Inputs: `assets/aerial/park.jpg`, `terrain.jpg`, and
`reference_annotations.coco.json`. Outputs: `docs/media/reference_park.jpg`,
`reference_terrain.jpg`, and `REFERENCE_PREVIEWS.json`. These are **reference
annotations, not model output**; all 16 boxes come from the source export.

After running the image and synthetic video/tracking demos above:

```bash
.venv-level-b/bin/python scripts/render_demo_previews.py
```

Inputs: the two included aerial photos, `outputs/aerial_park_detected.jpg`,
`outputs/aerial_terrain_detected.jpg`,
`outputs/synthetic_civilian_input.mp4`, and `outputs/synthetic_civilian_tracked.mp4`.
Outputs: `docs/media/aerial_samples.jpg`, `docs/media/detection.jpg`,
`docs/media/detection_2.jpg`, and `docs/media/tracking.gif`. These small
publication assets are included in Git; raw runs under `outputs/` are ignored.

To rebuild the two sample JPEGs and their provenance manifest from the optional
local export, run from the repository root:

```bash
.venv-level-b/bin/python scripts/prepare_aerial_samples.py
```

This regeneration step needs `Heridal.v2-yolo_non_412.coco/`. Ordinary demo and
preview commands use the included samples and do not need the full dataset.

## Publication Check

From the repository root, with Git initialized:

```bash
.venv-level-b/bin/python scripts/verify_publication.py --smoke
```

This checks publishable contents and links, then runs tests and synthetic CLI
demos in a temporary copy without local datasets or pretrained weight files.
It uses existing environment dependencies; it does not install a fresh environment.

## New Training CLI

The separate `aerial-train` entry point provides `prepare`, `train`, `evaluate`,
and `infer`. It requires `requirements-training.txt` for COCO evaluation. Exact
local, resume, checkpoint-inference, and Colab instructions are in
[TRAINING.md](TRAINING.md). These are new offline experiments beyond Level B.

## Optional Dependencies And Limits

`dev` provides pytest. `legacy-goturn` declares OpenCV contrib but cannot enable
the stub. The `ros` extra is empty and does not install ROS2. Webcam input needs
OpenCV and was not tested. No optional extras are needed for the standard demos.
The local Heridal export is not required for the original demos, but is required
for the new training experiments. See [dataset status](DATASET_STATUS.md).
