# Aerial Object Detection: Project Status

Reviewed on October 22, 2026. Commands run from the repository root.

Public repository: `aerial-object-detection`. Neutral entry points are
`aerial-detect` and `aerial-train`; older aliases/imports remain compatible.
Main README previews now show explicitly labeled source annotations, not model
predictions. See [publication instructions](docs/GITHUB_HANDOFF.md).

| Area | Status |
|---|---|
| COCO utility | Implemented and tested |
| Reference annotation previews | Implemented and tested; dataset boxes explicitly labeled as not model output |
| Public pretrained image detector | CPU execution tested on two HERIDAL frames; 0 predictions at 0.50 |
| Video pipeline / IoU tracker | Implemented and tested with synthetic input and mock detector |
| Mock detector / mock tracker | Mock only, tested as software fixtures |
| ROS2 helpers | Tested without a ROS graph |
| ROS2 runtime | Blocked by missing dependencies; actual node untested |
| GOTURN | Stub only; cannot run even after installing the optional extra |
| Webcam / real-detector video | Implemented but untested end to end |
| Offline training | Implemented and tested on CPU: 535 images, 1 epoch, 1,526 optimizer steps; checkpoint reload/resume verified |
| Colab training | Notebook prepared; cloud execution not performed |
| Drone control / navigation / deployment | Not implemented |
| GitHub Actions | Workflow configured; hosted execution pending publication |

macOS arm64 CPU operation was validated with Python 3.11.11. CUDA is not
required for the verified demos. Python 3.12 and Linux CI are not yet validated
in their target environments.

## First Demo

After installation, from the repository root:

```bash
.venv-level-b/bin/aerial-detect run-pipeline --config configs/synthetic_video_demo.toml
```

Output: `outputs/synthetic_civilian_tracked.mp4`. No external input is needed.

## Handoff

- [Exact installation, test, and demo commands](docs/RUNBOOK.md)
- [Detailed original handoff](docs/PROJECT_STATUS.md)
- [Aerial sample update and verification](docs/validation/AERIAL_SAMPLES.md)
- [Training commands and local/Colab scope](docs/TRAINING.md)
- [New training execution record](docs/validation/TRAINING_VALIDATION.md)
- [Previous publication preparation](docs/validation/GITHUB_PREPARATION.md)
- [Level A record](docs/validation/LEVEL_A.md) and [Level B record](docs/validation/LEVEL_B.md)
- [Historical vs reconstructed](docs/HISTORICAL_VS_RECONSTRUCTED.md)
- [Local dataset status](docs/DATASET_STATUS.md)
- [GitHub publication instructions](docs/GITHUB_HANDOFF.md)
- [Neutral naming and preview validation](docs/validation/PUBLICATION_REFRESH.md)

The archive provided evidence and ideas; no historical implementation is copied
into this package. No original completed training or metrics are claimed. A local
Heridal export is present but unverified against the original data and ignored
by Git except for two small attributed aerial examples. Still-image demos now
use those frames; video/tracker demos still use generated synthetic footage.
New optional training uses the audited local export and stores ignored artifacts
under `outputs/training/`. Its metrics are new subset results, not historical ones.
The full-split run's mAP changed from 0.1151 to 0.1940 on just eight validation
images (11 boxes); this is not a full benchmark. False positives and misses
remain visible. Best weights: `outputs/training/local_full/best.pt` (local only).
