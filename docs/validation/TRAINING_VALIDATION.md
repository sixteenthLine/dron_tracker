# New Offline Training Validation

Run date: October 22, 2026. This records new reconstruction work, not recovered
historical results. All shell commands below run from the repository root.
The local dataset, logs, and checkpoints remain ignored by Git.

## Environment And Installation

macOS, Apple M2, 8 GiB memory, Python 3.11.11 in the existing isolated
`.venv-level-b` environment; Torch 2.5.1 and torchvision 0.20.1. No CUDA hardware.
The existing environment was reused rather than claiming a fresh reinstall.

```bash
.venv-level-b/bin/python -m pip install -r requirements-training.txt
.venv-level-b/bin/python -m pip install -e . --no-deps
.venv-level-b/bin/python -m pip check
```

All three commands succeeded. `pycocotools==2.0.11` was added to the existing
environment; the editable package was reinstalled with the new `falcon-train`
entry point. The final dependency check reported `No broken requirements found`.

## Data

The full local export was audited: 1,546 images, 2,892 boxes, no missing images,
invalid boxes, or exact SHA-256 duplicates. 561 unannotated images were excluded,
not assumed verified negatives. Training/validation/test splits contain
535/167/283 positive images and 1,152/267/1,473 boxes. Prefixes are scene proxies,
not verified geographic independence. Annotation completeness and near duplicates
remain unverified. The test split has not been benchmarked.

Executed successfully:

```bash
.venv-level-b/bin/falcon-train prepare --dataset Heridal.v2-yolo_non_412.coco --output outputs/training/data > outputs/training/prepare_console.log 2>&1
```

Repeated preparation produced the same manifest SHA-256 used by the checkpoint:
`cf561774a0d3156b483d41905fde1586b29cdc2b5af44e50f7ff5a8210e4e171`.
Audit/split records are in `outputs/training/data/audit.json` and `manifest.json`.

## Successfully Executed Training

Model: torchvision `fasterrcnn_mobilenet_v3_large_320_fpn`, public DEFAULT
weights, new two-class predictor initialized from background/person rows.
Local input is 512 pixels after native 768-pixel tiling with 128-pixel overlap.
ROI heads only were updated: 13,905,930 trainable parameters. Backbone and RPN
were frozen. Initial/final checkpoint comparison found eight changed model
tensors, all under `roi_heads`, confirming real parameter updates.

```bash
.venv-level-b/bin/python -m falcon_vision_reconstructed.training train --config configs/training_local_pilot.toml
.venv-level-b/bin/python -m falcon_vision_reconstructed.training train --config configs/training_local.toml > outputs/training/local_full_console.log 2>&1
```

| Run | Training images / positive tiles | Optimizer steps | Validation images / boxes | Elapsed | Initial mAP | Final mAP |
|---|---|---|---|---|---|---|
| Bounded pilot | 64 / 182 | 120 | 2 / 3 | 53.40 s | 0.05610561 | 0.06732673 |
| Full training split, one epoch | 535 / 1,526 | 1,526 | 8 / 11 | 544.31 s | 0.11514143 | 0.19403631 |

Both exited 0. Pilot stop reason: `step_limit`; full-split stop reason:
`epochs_complete`. Mean losses were 0.37861457 and 0.23143309 respectively.
The full-split AP50 changed from 0.25751259 to 0.41114886, and AP75 from
0.03300330 to 0.18389168. These are new small-subset observations, not a
convergence claim, test benchmark, or historical metric.

Each run saved `baseline.pt`, `best.pt`, `last.pt`, `baseline_metrics.json`,
`config.json`, `history.jsonl`, and `summary.json` beneath its own directory:
`outputs/training/local_pilot/` or `outputs/training/local_full/`.
The full-split `best.pt` and `last.pt` each occupy 131,624,813 bytes and record
step 1,526. They are local only and not part of a fresh source checkout.

## Successfully Executed Reload And Resume

```bash
.venv-level-b/bin/python -m falcon_vision_reconstructed.training train --config configs/training_local_pilot.toml --resume outputs/training/local_pilot/last.pt --max-steps 121 --output-dir outputs/training/resume_smoke > outputs/training/resume_smoke_console.log 2>&1
.venv-level-b/bin/falcon-train evaluate --checkpoint outputs/training/local_full/best.pt --manifest outputs/training/data/manifest.json --image-root Heridal.v2-yolo_non_412.coco/train --split val --limit 8 --device cpu --output outputs/training/local_full/rechecked_val.json > outputs/training/recheck_console.log 2>&1
```

Both exited 0. Real-model resume restored step 120 and executed one additional
optimizer step in 14.85 s. Step 121's model differs from step 120; the original
pilot run was not overwritten. New state/logs are under
`outputs/training/resume_smoke/`. It contains `last.pt`, not a new best checkpoint
because that extra step did not beat the previous best AP.

Reloaded-checkpoint evaluation exactly matched the full run's final metrics on
image IDs `[956, 306, 212, 1018, 68, 974, 1214, 218]`. Rebuilding preserves the
pretrained backbone's 46 FrozenBatchNorm layers (zero trainable BatchNorm layers).
No historical checkpoint was involved.

## Successfully Executed Image Inference

```bash
.venv-level-b/bin/python -m falcon_vision_reconstructed.training infer --checkpoint outputs/training/local_pilot/best.pt --input assets/aerial/park.jpg --output outputs/training/pilot_park_preview.jpg --device cpu --threshold 0.50
.venv-level-b/bin/falcon-train infer --checkpoint outputs/training/local_full/best.pt --input assets/aerial/park.jpg --output outputs/training/local_full/park_preview.jpg --device cpu --threshold 0.50
.venv-level-b/bin/falcon-train infer --checkpoint outputs/training/local_full/best.pt --input Heridal.v2-yolo_non_412.coco/train/train_ZRI_2063_JPG.rf.fbbd84ecf8857e3d62fe6aabce349d17.jpg --output outputs/training/local_full/park_native_preview.jpg --device cpu --threshold 0.50
```

All exited 0 and generated images. The pilot preview has 13 predicted boxes.
Full-run resized preview: 1,600 x 1,200, 217,588 bytes, 5 predicted boxes.
Native preview: 4,000 x 3,000, 1,343,540 bytes, 21 predicted boxes.
These counts are **not counts of correct detections**. Visual inspection found
misses and false positives; the native frame finds more visible people but also
labels some objects/vegetation incorrectly. Resizing changes object scale.
This visual preview of a reserved test frame is not a quantitative test benchmark.

## Software And Publication Checks

```bash
.venv-level-b/bin/falcon-train --help
.venv-level-b/bin/python -m pytest -q
.venv-level-b/bin/python scripts/verify_publication.py --smoke
.venv-level-b/bin/python scripts/build_training_bundle.py
```

The CLI exposes `prepare`, `train`, `evaluate`, and `infer`. All 34 tests pass;
the final direct pytest run completed in 2.43 s.
The 13 new training tests cover group separation, duplicate rejection, tiling,
coordinate conversion, native pixel scale, deterministic subset selection,
perfect/empty COCO results, real optimizer/checkpoint/resume on a tiny fixture,
resume learning-rate mismatch rejection, config validation, frozen normalization,
and notebook source syntax. Tests download neither data nor model weights.

The notebook JSON and its six code cells parse locally, have null execution
counts, and contain no recorded outputs. This is source validation only.
The workflow also passed local Actionlint validation; hosted GitHub CI is pending.
Publication smoke checks passed for all 66 publishable files, including links,
exclusions, tests, and synthetic CLI commands in a temporary source-only copy
without the local dataset or cached weight files. The Colab ZIP is
`outputs/falcon_training_project.zip`; it excludes datasets, checkpoints,
environments, and logs. Nothing was uploaded, committed, or pushed.

## Device Failure

MPS passed a dummy forward/backward but stalled during real tiled validation
in Metal/nonzero synchronization. That attempt was terminated before training
updates. It is not a successful MPS training run. CPU is the validated local
route; `auto` uses CUDA when available, otherwise CPU.

## Interpretation

Metrics are COCO bbox AP on a small positive-only validation subset. They do not
establish full-dataset quality, background false-positive rates, safety, or
deployment readiness. Initial and updated models use identical binary heads,
resolution, fixed full-frame tiling, and evaluation. The legacy 320-pixel demo
has a different preprocessing protocol and is not a fair fine-tuning baseline.

## Not Executed

- Google Colab/Drive authorization, CUDA training, and the 20-epoch Colab run.
- Full 167-image validation or 283-image test benchmark.
- Training with all available backbone stages, multi-epoch convergence study,
  negative-image label review, or hyperparameter search.
- Any new tracking, video, ROS2, or drone integration with trained weights.

The notebook is a prepared workflow, not evidence of successful cloud execution.
Checkpoints are new experiment artifacts, not recovered historical model weights.
