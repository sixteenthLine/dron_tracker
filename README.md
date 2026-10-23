# Aerial Object Detection

A Python codebase for experimenting with object detection in civilian aerial imagery. The idea is to connect COCO-style data preparation, torchvision Faster R-CNN, tiled inference, and offline fine-tuning in a modular project. This repository contains experimental software, not a ready-to-use trained model or an onboard drone system.

The architecture separates dataset validation, frame sources, detector and tracker interfaces, visualization, and output writing. Image and video sources feed a configurable pipeline that produces annotated outputs and structured logs; TOML files define its components and settings. Training and checkpoint utilities form a separate offline workflow, while synthetic fixtures support testing. The detailed component and data-flow diagrams are in [System Architecture](docs/SYSTEM_ARCHITECTURE.md).
