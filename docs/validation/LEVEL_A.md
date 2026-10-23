# Level A Validation (Historical Record)

This is the original reconstruction validation record from August 2, 2026.
The former smoke-test photo is kept locally but excluded from publication
because its redistribution license was not established. Current commands and
the licensed replacement sample are documented in [the handoff](../PROJECT_STATUS.md).

Validation date: 2026-08-02

All commands were run from the `falcon-vision-reconstructed` directory.

## Environment

Command:

```bash
python3.11 --version
```

Result:

```text
Python 3.11.11
```

## Fresh Environment

Command:

```bash
python3.11 -m venv .venv
```

Result: completed successfully.

Command:

```bash
.venv/bin/python -m pip install --upgrade pip
```

Result: completed successfully; pip was upgraded to 26.2.

Command:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Result: completed successfully with pinned dependencies installed or already satisfied.

Command:

```bash
.venv/bin/python -m pip install -e .
```

Result: completed successfully; `falcon-vision-demo` console command was installed.

## Sample Image Metadata

Command:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from PIL import Image
path = Path('assets/sample_civilian_dog.jpg')
with Image.open(path) as image:
    image.convert('RGB').save(path, format='JPEG', quality=92, optimize=True)
PY
file assets/sample_civilian_dog.jpg
```

Result:

```text
assets/sample_civilian_dog.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1546x1213, components 3
```

## Tests

Command:

```bash
.venv/bin/python -m pytest
```

Result:

```text
collected 5 items
tests/test_coco_dataset.py ..... [100%]
5 passed in 1.62s
```

## Demo

Command:

```bash
.venv/bin/falcon-vision-demo \
  --input assets/sample_civilian_dog.jpg \
  --output outputs/sample_civilian_dog_detected.jpg \
  --threshold 0.50
```

Result:

```text
Saved outputs/sample_civilian_dog_detected.jpg with 1 detections at threshold 0.50 using torchvision Faster R-CNN MobileNet V3 Large 320 FPN, COCO pretrained weights.
```

Note: the first demo run downloaded public torchvision model weights into the local Torch cache.

## Output Artifact

Command:

```bash
test -s outputs/sample_civilian_dog_detected.jpg && file outputs/sample_civilian_dog_detected.jpg && ls -lh outputs/sample_civilian_dog_detected.jpg
```

Result:

```text
outputs/sample_civilian_dog_detected.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1546x1213, components 3
-rw-r--r--  1 user  staff   264K outputs/sample_civilian_dog_detected.jpg
```

## Scope Check

No training, tracking, video processing, ROS integration, drone integration, ONNX export, Docker work, or Level B work was performed.

