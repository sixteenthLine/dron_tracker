# Sanitized Historical Audit Summary

This document copies the relevant audit findings in sanitized form. It removes absolute local paths, personal filesystem details, and operationally sensitive implementation details.

This is the original audit perspective. A local Heridal export is now present;
its relationship to the historical dataset remains unverified. See
[current dataset status](DATASET_STATUS.md).

## Executive Summary

The recovered workspace was not a complete runnable project. It contained two main historical tracks:

- a FalconVision detector/data-loading track with COCO-style dataset code and Faster R-CNN notebook experiments;
- a separate OpenCV/GOTURN tracking branch plus drone-adjacent ROS material.

The detector evidence supports a historical prototype, not a finished trained system. Saved notebook output showed that dataset batches loaded and that detector training began, but training was interrupted and no completed detector checkpoint or verified historical metrics were recovered.

The audit did not establish that detector, tracking, and ROS pieces were ever integrated into a complete system.

## Preserved Detector-Relevant Files

- `FalconVision/Dataprep.py`: custom dataset-loader idea.
- `FalconVision/ReadMe.md`: loader usage notes.
- `FalconVision/FalconDataloader.ipynb`: early exploratory data-loader notebook.
- `FalconVision/HeridalDataset.ipynb`: later detector notebook with dataset loading, Faster R-CNN setup, training-loop code, evaluation-loop code, and saved interrupted output.
- `FalconVision/test.py`: small loader smoke-test script with hard-coded paths removed from this summary.

## Preserved Tracking-Relevant Evidence

- An OpenCV motion/tracking branch used frame differencing and a GOTURN-style tracker handoff.
- GOTURN Caffe model/prototxt assets existed in the archive.
- A vendored third-party GOTURN-PyTorch repository was present.

The tracking branch referenced local video inputs that were not present. No project-specific successful tracking output or trained project checkpoint was recovered.

## Preserved ROS/Drone-Adjacent Evidence

- A manual ROS teleoperation artifact existed in the archived material.
- No confirmed historical computer-vision-to-ROS integration was found.

The teleoperation material is excluded from reconstruction. It is not reused in Level B.

## What Appeared To Work

- COCO-style JSON parsing was implemented in several forms.
- Images and target dictionaries were produced in saved notebook output.
- Saved output showed image tensors shaped like high-resolution RGB images.
- Faster R-CNN was instantiated in a notebook.
- A training loop started and advanced into epoch 0.

## What Was Incomplete Or Missing

- The referenced dataset and annotation file were not present in the audited workspace.
- No trained detector checkpoint was recovered.
- No completed mAP, loss curve, or benchmark result was recovered for the FalconVision detector.
- No standalone, reproducible detector inference command was found.
- No clean dependency file for the detector work was found.
- No output video writer, safe demo video, or finished tracker demo output was recovered.
- No evidence showed that the detector, tracker, and ROS material were historically integrated.

## Excluded Material

The audit also found drone-control and operationally sensitive fragments. Those files are not used in this reconstruction. The reconstruction excludes:

- real or simulated drone control;
- navigation or autonomous behavior;
- target following;
- target selection or prioritization;
- engagement, trigger, actuator, or mission logic;
- operational surveillance workflows.

## Safe Reconstruction Conclusion

A safe portfolio demo is realistic if it is described as new reconstruction code based on preserved historical ideas. The honest scope is:

- preserve the dataset-loader concept;
- use public pretrained detector weights for image inference;
- run video/tracking demos on synthetic safe footage;
- keep ROS2 optional and limited to camera input plus perception outputs;
- avoid claims of historical training completion, historical performance, or historical integration.
