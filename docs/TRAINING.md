# Offline Training Experiments

Current public CLI: `aerial-train`, distribution: `aerial-object-detection`.
The older alias/import namespace is retained for saved experiments. Dated
execution records show the commands actually used before this naming change.

This is new reconstruction work requested after Level B. Historical folders
remain unchanged, and no historical completed training or metrics are claimed.
The original image/video demos retain their public pretrained detector; optional
training commands create separate binary human/background checkpoints.

## Data Audit And Split

The local export contains 1,546 images and 2,892 category-1 boxes. The preparation
pass checked image presence, header dimensions, annotation bounds, and exact
SHA-256 duplicates. No missing images, invalid boxes, or exact duplicates were
found. It does not prove complete labeling or detect all visually similar frames.

The 561 unannotated images are excluded until reviewed. They are not silently
treated as verified negatives. This yields a **positive-only training/evaluation
protocol**, so reported validation metrics cannot establish false-positive
performance on the full scene distribution.

| Split | Filename groups | Images | Boxes |
|---|---|---|---|
| Train | BLA, BRA, BRS, CAP, GOR, JAS, MED, SB, TRS | 535 | 1,152 |
| Validation | BRK, RAK | 167 | 267 |
| Test | VRD, ZRI | 283 | 1,473 |

Filename prefixes are a scene proxy, not verified geographic independence. No
prefix crosses splits, and exact duplicate hashes cannot cross splits. The
README's park/terrain frames belong to the reserved test groups. Training crops
are generated **after** this split. Test metrics are not used for model selection.

## Model And Resolution

The experiment reuses torchvision Faster R-CNN MobileNet V3 Large FPN's public
weights. Its binary predictor starts from COCO's background/person classifier
and box-regression rows. A fixed 768-pixel grid with 128-pixel overlap operates
on native-resolution images, rather than first shrinking an entire 4000 x 3000
frame to 320 pixels. Local model input is 512 pixels; Colab configuration uses 768.

Local runs update ROI heads with the backbone/RPN frozen. The Colab configuration
updates all parts enabled by the pretrained torchvision factory, including its
trainable backbone stages. This is fine-tuning, not training from scratch.

Validation uses the same fixed full-frame grid without annotation-centered
windows, restores original coordinates, applies global torchvision NMS, and
uses **pycocotools COCOeval**. The initial and fine-tuned binary models use the
same resolution/tiling, so their comparison isolates weight updates. This differs
from the old 320-pixel, 91-class demo, so improvement must not all be attributed
to fine-tuning when comparing against that older demo.

## Installation And Commands

Run all local commands from the repository root:

```bash
.venv-level-b/bin/python -m pip install -r requirements-training.txt
.venv-level-b/bin/python -m pip install -e . --no-deps
.venv-level-b/bin/aerial-train prepare --dataset Heridal.v2-yolo_non_412.coco --output outputs/training/data
```

If the environment is not activated, use `.venv-level-b/bin/aerial-train` or
`.venv-level-b/bin/python -m falcon_vision_reconstructed.training` in place of
`aerial-train` for every command below.

Bounded pilot (64 images, up to 120 steps, two validation images):

```bash
.venv-level-b/bin/aerial-train train --config configs/training_local_pilot.toml
```

One pass over the complete audited training split (535 images; frozen-backbone
head fine-tuning, eight validation images, 10-minute training-loop time budget):

```bash
.venv-level-b/bin/aerial-train train --config configs/training_local.toml
```

Existing run folders cannot be silently overwritten. To continue a pilot to a
higher total step count, run from the root:

```bash
.venv-level-b/bin/aerial-train train --config configs/training_local_pilot.toml --resume outputs/training/local_pilot/last.pt --max-steps 121
```

Resume retains optimizer state, sampling progress, and RNG states and verifies
the manifest hash plus shape/sampling settings. To continue after a completed
epoch limit, increase `epochs` in a new config that otherwise matches the run.
Learning-rate changes require a new run, rather than silently overriding the
restored optimizer. Results need not be bit-for-bit identical across devices;
CUDA operator nondeterminism can also affect repeated runs. In-run checkpoints are
atomic local replacements; an interruption can lose steps since the last save.

Evaluate a saved checkpoint on a fixed validation subset:

```bash
.venv-level-b/bin/aerial-train evaluate --checkpoint outputs/training/local_full/best.pt --manifest outputs/training/data/manifest.json --image-root Heridal.v2-yolo_non_412.coco/train --split val --limit 8 --device cpu --output outputs/training/local_full/rechecked_val.json
```

Omit `--limit` for the entire selected split. Use `--split test` only after fixing
model settings; this is a separate final evaluation, not part of training.

Offline still-image inference:

```bash
.venv-level-b/bin/aerial-train infer --checkpoint outputs/training/local_full/best.pt --input assets/aerial/park.jpg --output outputs/training/local_full/park_preview.jpg --device cpu --threshold 0.50
```

The published JPEG is resized, so its preview is not a full-resolution test
benchmark. It also changes the apparent object scale relative to training.
For a native-resolution preview of the same frame, run from the root:

```bash
.venv-level-b/bin/aerial-train infer --checkpoint outputs/training/local_full/best.pt --input Heridal.v2-yolo_non_412.coco/train/train_ZRI_2063_JPG.rf.fbbd84ecf8857e3d62fe6aabce349d17.jpg --output outputs/training/local_full/park_native_preview.jpg --device cpu --threshold 0.50
```

This requires the full local export, not just the included JPEG. Human class
boxes do not recognize an individual's identity. Both preview protocols still
show false positives and misses; a returned box count is not a count of correct
human detections.

## Artifacts And Evidence

Run artifacts are ignored by Git under `outputs/training/`:

- `data/audit.json`, `data/manifest.json`: audit and exact split membership.
- `local_pilot/` and `local_full/`: `config.json`, `history.jsonl`, `summary.json`.
- `baseline.pt`, `best.pt`, `last.pt`: initial, best-validation, and latest states.
- `baseline_metrics.json`: the resolution-matched initial model's validation result.

`best.pt` can remain the initial baseline if validation AP does not improve;
`last.pt` contains the latest actual updates. Checkpoints load with
`torch.load(..., weights_only=True)` and are not uploaded or redistributed.
Load only trusted checkpoints produced by your own runs; `weights_only` is not
a guarantee that arbitrary third-party checkpoint files are safe.
Current executions and results are recorded in
[TRAINING_VALIDATION.md](validation/TRAINING_VALIDATION.md).

## Colab

From the local repository root:

```bash
.venv-level-b/bin/python scripts/build_training_bundle.py
```

Open [Aerial_Detection_Training.ipynb](../notebooks/Aerial_Detection_Training.ipynb)
in Colab and select a GPU runtime. The first cell uploads the small
`outputs/aerial_detection_training.zip`; it contains source and public samples,
not the full dataset or weights. Put your dataset folder or archive in your
Drive and adjust the notebook paths. Drive access requires your authorization.

The notebook creates an isolated Python 3.11 environment, installs the pinned
CUDA 12.1 wheels and training dependencies, audits data, and starts/resumes
`configs/training_colab.toml`. Its default is 20 epochs over the train split,
with validation on eight fixed images and early stopping. CUDA execution and
Google account authorization were not performed locally; notebook source was
syntax-checked only. Free GPU resources and session duration are not guaranteed.

## Local Device Limits

CPU execution is validated. An MPS dummy forward/backward passed, but actual
full-frame validation stalled in Metal/nonzero synchronization and was stopped
before training updates. `auto` therefore selects CUDA when available, otherwise
CPU; MPS is opt-in and not advertised as a working training route.

The time budget is checked between training batches. It does not interrupt an
in-flight operator or final evaluation, so it is not a strict wall-clock kill
timer. Checkpoints, training data, and logs are kept local unless the user
explicitly uploads them. No ROS, navigation, drone control, or deployment is added.

## References

- [Official torchvision fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
- [Official Torch 2.5.1 CUDA wheels](https://pytorch.org/get-started/previous-versions/).
- [Colab resource limits](https://research.google.com/colaboratory/faq.html).
