# Historical Evidence

This document summarizes the audit findings relevant to the Level B reconstruction. Historical paths have been sanitized to avoid absolute local paths and personal filesystem details.

Dataset absence below refers to the original audit. The current local export
and its unresolved historical identity are described in [DATASET_STATUS.md](DATASET_STATUS.md).

## Detector Branch

Evidence:

- `FalconVision/Dataprep.py` contained a COCO-style dataset loader idea.
- `FalconVision/HeridalDataset.ipynb` instantiated a torchvision Faster R-CNN style detector.
- Notebook output showed that training began.

Status:

- The referenced dataset and annotation files were missing.
- Training output was interrupted during epoch 0.
- No completed checkpoint, model card, or historical metric output was recovered.

Level B use:

- The dataset-loader idea was rewritten as `CocoDetectionDataset`.
- The detector factory uses modern public torchvision pretrained weights.
- No historical training completion or historical performance is claimed.

## Tracking Branch

Evidence:

- A separate OpenCV branch used frame differencing and GOTURN-style tracking experiments.
- GOTURN Caffe model/prototxt files existed in the historical archive.
- A vendored third-party GOTURN-PyTorch repository was present.

Status:

- No saved successful project-specific tracking output was recovered.
- The referenced local video was missing.
- The third-party GOTURN branch did not provide a project-specific trained checkpoint.
- The GOTURN model files need provenance and license review before redistribution.

Level B use:

- GOTURN is documented as historical reference only.
- A safe `IoUTracker` was implemented as new work.

## ROS/Drone-Adjacent Material

Evidence:

- A manual ROS teleoperation artifact existed.
- The audit found no confirmed CV-to-ROS integration.

Status:

- The teleoperation material is excluded from this safe reconstruction.
- No historical deployment or integrated system is claimed.

Level B use:

- A new optional ROS2 camera CV bridge was written with perception output topics only.
- It publishes detections, tracks, debug image, and status.
- It does not publish movement, navigation, mission, actuator, target-selection, or control commands.
