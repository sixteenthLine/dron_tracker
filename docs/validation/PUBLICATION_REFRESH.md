# Neutral Naming And Reference Preview Validation

October 22, 2026. Commands run from the repository root. Historical folders,
training weights, and the original model-output images were not modified.

## Naming

Public title/distribution/repository: **Aerial Object Detection** /
`aerial-object-detection`. New commands: `aerial-detect`, `aerial-train`.
Existing aliases and `falcon_vision_reconstructed` imports remain compatible.
The current local folder was not renamed because the virtual environment and
saved local links contain its path. Names in dated historical records remain
evidence, not current branding or a claim of affiliation with the former team.

The Colab notebook is now `notebooks/Aerial_Detection_Training.ipynb`.
The source bundle is `outputs/aerial_detection_training.zip` with a neutral
`aerial-object-detection/` root. Cloud execution is still untested.

## Reference Images

```bash
.venv-level-b/bin/python scripts/prepare_aerial_samples.py
.venv-level-b/bin/python scripts/render_reference_previews.py
```

Both succeeded. The first reads the existing civilian export without changing
it and writes only the two sample JPEGs, provenance, and their 16-box COCO subset.
Sample image hashes still match the preceding publication evidence. Coordinates
are scaled, not invented. The second uses only the included sample files and
annotation subset; it does not run a detector or download data/weights.

Outputs: `docs/media/reference_park.jpg` (15 boxes) and
`reference_terrain.jpg` (1 box), both 1280 x 1152. Each image has a visible
**REFERENCE ANNOTATIONS / NOT MODEL OUTPUT** header. Numbered boxes have no
fabricated confidence scores. Source/output hashes are recorded in
`docs/media/REFERENCE_PREVIEWS.json`. Both previews were visually inspected.
The README distinguishes these reference layouts from actual measured results.

## Environment And Tests

```bash
.venv-level-b/bin/python -m pip uninstall -y falcon-vision-reconstructed
.venv-level-b/bin/python -m pip install -e . --no-deps
.venv-level-b/bin/python -m pip check
.venv-level-b/bin/python -m pytest -q
.venv-level-b/bin/aerial-detect validate
.venv-level-b/bin/aerial-train --help
.venv-level-b/bin/falcon validate
.venv-level-b/bin/python scripts/verify_publication.py --smoke
```

The existing isolated environment was reused. The old editable distribution was
replaced with `aerial-object-detection==0.1.0`; the neutral and legacy commands
are present. Dependency check: `No broken requirements found`. All 37 tests
passed, including three new reference-preview tests. An initial test invocation
started before reinstall finished and failed on imports; the post-install rerun
passed. Publication checks passed in a temporary source-only copy, including
links, sample/annotation/preview hashes, tests, and synthetic CLI demos.
Availability checks still report ROS2 absent; no actual ROS2 runtime is claimed.

Actual inference was also repeated successfully through the neutral entry points:

```bash
.venv-level-b/bin/aerial-detect infer-image --input assets/aerial/park.jpg --output outputs/publication_check/neutral_baseline.jpg --threshold 0.50
.venv-level-b/bin/aerial-train infer --checkpoint outputs/training/local_full/best.pt --input assets/aerial/park.jpg --output outputs/publication_check/neutral_finetuned.jpg --device cpu --threshold 0.50
.venv-level-b/bin/python scripts/build_training_bundle.py
```

The baseline generated an image with zero predictions; the step-1,526 checkpoint
generated one with five predictions. These counts are not reference-box counts
or counts of correct detections. No new training was run. The rebuilt ZIP passed
CRC/content checks and excludes the dataset, model weights, and environments.
The current workflow passed local Actionlint; hosted execution is still pending.

## Publication Boundary

`git add --dry-run .` was inspected and excluded the full dataset, checkpoints,
environments, and ignored outputs. Git staging, commits, remote creation, and
push were not performed. GitHub-hosted CI is pending. Exact publication commands
are in [GITHUB_HANDOFF.md](../GITHUB_HANDOFF.md).
