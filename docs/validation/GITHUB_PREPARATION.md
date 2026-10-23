# GitHub Preparation Validation

This record describes the earlier photo-based preparation pass. Its image,
preview sizes, and publication counts were superseded by
[the aerial sample update](AERIAL_SAMPLES.md). Original results below are retained.

Date: October 22, 2026. All project commands run from the repository root.
Environment: macOS arm64, Python 3.11.11, existing `.venv-level-b`, CPU.
This pass reused the previously validated environment; it did not create a new
one or validate Linux/Python 3.12. Source modules under `src/` were unchanged
(SHA-256 baseline compared before/after); historical archive files were not edited.

## Executed Commands

| Exact command | Result |
|---|---|
| `.venv-level-b/bin/python -m pip install -e . --no-deps` | Passed; editable metadata refreshed |
| `.venv-level-b/bin/python -m pip check` | Passed; no broken requirements |
| `.venv-level-b/bin/python -m pytest -q tests/test_coco_dataset.py tests/test_tracking.py tests/test_ros_bridge.py` | 16 passed |
| `.venv-level-b/bin/python -m pytest -q tests/test_pipeline.py` | 5 passed |
| `.venv-level-b/bin/falcon infer-image --input assets/sample_public_dog.jpg --output outputs/public_dog_detected.jpg --threshold 0.50` | Passed; 1 detection |
| `.venv-level-b/bin/falcon-vision-demo --input assets/sample_public_dog.jpg --output outputs/public_dog_detected_alias.jpg --threshold 0.50` | Passed; 1 detection |
| `.venv-level-b/bin/falcon create-demo-video --output outputs/synthetic_civilian_input.mp4 --frames 24 --fps 12` | Passed; 24 frames |
| `.venv-level-b/bin/falcon run-pipeline --config configs/synthetic_video_demo.toml` | Passed; 24 frames, one mock detection / IoU track per frame |
| `.venv-level-b/bin/falcon infer-video --input outputs/synthetic_civilian_input.mp4 --output outputs/synthetic_civilian_infer_video.mp4 --detector mock --tracker iou --max-frames 8 --fps 12` | Passed; 8 frames |
| `.venv-level-b/bin/falcon ros-node --check-only` | Passed; `ros2_available: false` |
| `.venv-level-b/bin/falcon validate` | Passed; availability/topic JSON only |
| `.venv-level-b/bin/python scripts/render_demo_previews.py` | Passed; JPEG and 24-frame GIF |
| `git init -b main` | Passed; local repository initialized |
| `git check-ignore .venv-level-b/ Heridal.v2-yolo_non_412.coco/ outputs/ assets/sample_civilian_dog.jpg .DS_Store example.pth .env` | Passed; all excluded |

## Workflow Validation

Actionlint **1.7.12** was downloaded from its
[official release](https://github.com/rhysd/actionlint/releases/tag/v1.7.12)
into a temporary directory. The archive SHA-256 was matched to the release
checksums before execution. From the repository root:

```bash
/tmp/falcon-actionlint.aVe4XN/actionlint -color .github/workflows/tests.yml
```

Result: exit 0, no findings. The temporary tool is not a project dependency or
publication file. Workflow references pin verified `actions/checkout` and
`actions/setup-python` v6 commits. **GitHub-hosted execution is still pending**;
static validation does not establish that the Linux dependency install works.

## Publication Verification

From the repository root:

```bash
.venv-level-b/bin/python scripts/verify_publication.py --smoke
git add --dry-run .
```

The verifier checks publication candidates, local Markdown links, file sizes,
and excludes local data/weights/environments. With `--smoke` it copies only
publishable files into a temporary directory, runs the 21 tests and synthetic
CLI demos, and checks the three generated videos. It uses the existing Python
environment's dependencies, an empty Torch cache, and the copied source tree;
it is not a fresh dependency installation or a GitHub runner.

Observed result: **passed**. Publication candidates: **49 files, approximately
550 KB**. All local links resolved and local-only files were excluded. The clean
copy passed all 21 tests and the synthetic commands, producing three nonempty
MP4s. The first-commit dry run listed only the intended publication files.

## Media Checks

- Input: `assets/sample_public_dog.jpg`, 97,307 bytes, 768 x 1024.
- SHA-256: `59b702079f42abfcdf630d1254a56c5861ba50dcf417d9e3b08cf8bfbbeda17b`.
- Image output: `outputs/public_dog_detected.jpg`, 101,762 bytes.
- Published image preview: `docs/media/detection.jpg`, 92,397 bytes.
- Published animation: `docs/media/tracking.gif`, 205,800 bytes, 664 x 284,
  24 frames; first/last frames differ. JPEG and GIF were visually inspected.
- MP4 frame counts checked by decoding: raw 24, tracked 24, direct inference 8.

## Not Executed Or Claimed

- No commit, remote creation, push, or GitHub-hosted workflow run.
- No actual ROS2 node startup or topic-delivery test; ROS2 is unavailable.
- No GOTURN execution: the wrapper is a stub even with optional dependencies.
- No webcam test, real-detector video demo, Python 3.12 run, training, model
  export, drone control, or deployment.
- No Heridal images used. Local export metadata is documented, not treated as
  proof of historical training data or completed training.
