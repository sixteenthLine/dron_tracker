# Aerial Sample Update

Date: October 22, 2026. All commands run from the repository root in the existing
Python 3.11.11 macOS arm64 CPU environment (`.venv-level-b`).

## Change

The published still-image demo now uses two full-frame photos from the local
HERIDAL Roboflow export, replacing the earlier dog photo. Original dataset files
remain unchanged. Only resized `assets/aerial/park.jpg` and `terrain.jpg` plus
their provenance manifest are included in Git; the full export remains ignored.
No detector, tracker, model weights, training, or runtime integration was changed.

The export's COCO license records and public project page declare CC BY 4.0.
Original HERIDAL authors are credited, and the upstream CC BY 3.0 distribution
notice is documented separately in [media provenance](../../assets/SAMPLE_SOURCE.md).

## Commands And Observed Results

```bash
.venv-level-b/bin/python scripts/prepare_aerial_samples.py
```

Result: two 1600 x 1200 full-frame JPEGs and a JSON provenance manifest. The
source frames are image IDs 0 and 100 from the local export. Source/sample
SHA-256 hashes are recorded. There is no crop, reference-box overlay, or training.

```bash
.venv-level-b/bin/falcon infer-image --input assets/aerial/park.jpg --output outputs/aerial_park_detected.jpg --threshold 0.50
.venv-level-b/bin/falcon infer-image --input assets/aerial/terrain.jpg --output outputs/aerial_terrain_detected.jpg --threshold 0.50
```

Both commands exited 0 and wrote images. **Each returned 0 detections at 0.50.**
This verifies inference execution and file output, not effective aerial detection.
The generic public COCO model is unchanged and no attempt was made to tune,
train, tile, or select crops to improve the result.

```bash
.venv-level-b/bin/falcon-vision-demo --input assets/aerial/park.jpg --output outputs/aerial_park_detected_alias.jpg --threshold 0.50
.venv-level-b/bin/python scripts/render_demo_previews.py
.venv-level-b/bin/python scripts/verify_publication.py --smoke
git check-ignore assets/sample_public_dog.jpg Heridal.v2-yolo_non_412.coco/ outputs/
```

The renderer uses the two aerial inputs, their actual inference outputs, and
the previously validated synthetic input/tracked videos. It creates the aerial
gallery, two input/output comparisons, and the synthetic GIF. The verifier checks
sample hashes, publication contents, links, and runs the 21 tests and synthetic
CLI demos in an isolated copy. It reuses installed dependencies and does not
require the full dataset or pretrained weights for the isolated smoke test.

Observed result: all four commands above exited 0. The alias also returned zero
boxes. Publication verification passed with **55 files, approximately 2.1 MB**;
sample hashes, exclusions, and local links passed. The isolated copy passed all
21 tests and produced the three synthetic MP4s. Source image SHA-256 hashes
still match the manifest; source implementation files under `src/` are unchanged.

Published inputs are 419,202 bytes (`park.jpg`) and 909,011 bytes (`terrain.jpg`).
The gallery and input/output comparisons are 984 x 404 pixels. The synthetic
GIF remains 664 x 284 with 24 frames. Aerial layouts were visually inspected.

## Limitations

- The two examples are demonstrations, not a representative evaluation split.
- No aerial detection accuracy, historical checkpoint, or completed historical
  training is claimed. Source dataset identity is still not proven against the
  exact historical notebook inputs.
- The video/tracker GIF remains synthetic and uses mock detections; these two
  still images are not presented as a real tracking sequence.
- ROS2 runtime, GOTURN, and webcam status are unchanged.
