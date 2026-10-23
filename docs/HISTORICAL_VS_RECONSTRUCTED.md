# Historical vs Reconstructed

These findings describe the original audit. A local Heridal export is now
present but has not been established as the exact historical training data;
see [DATASET_STATUS.md](DATASET_STATUS.md).

| component | historical evidence | historical status | current reconstruction | whether it is new work |
|---|---|---|---|---|
| detector | Notebook evidence showed a torchvision Faster R-CNN prototype and training-loop code. | Prototype only; training was interrupted and no checkpoint or metrics were recovered. | `TorchvisionDetector` wraps a modern public COCO-pretrained Faster R-CNN MobileNet V3 320 FPN model for safe inference demos. | Yes, new implementation using public weights. |
| dataset loader | `FalconVision/Dataprep.py` used COCO-style `images`, `annotations`, and `categories`. | Confirmed idea, but tied to missing local data and hard-coded paths. | `CocoDetectionDataset` validates JSON, supports configurable labels, converts `xywh` to `xyxy`, and handles empty boxes. | Yes, rewritten. |
| video pipeline | Historical files had notebook/webcam experiments and separate OpenCV tracking scripts. | No complete reusable video pipeline or output writer was recovered. | `FrameProcessor` connects sources, detector, tracker, overlay, sink, config, and structured logging. | Yes, entirely new. |
| GOTURN | Separate OpenCV/GOTURN experiments and model/prototxt assets were present. | Incomplete; no safe successful output recovered; dependency and model provenance need review. | `LegacyGoturnTracker` is an unavailable-by-default compatibility boundary and is not used by demos. | Yes, boundary wrapper only. |
| modern tracker | No modern generic multi-object tracker was recovered. | Missing. | `IoUTracker` assigns generic track IDs from detection overlap with configurable lifecycle. | Yes, entirely new. |
| ROS teleoperation artifact | A manual ROS teleoperation artifact was present in archived material. | Drone-control material, no confirmed CV integration. | Not reconstructed or imported. It remains excluded from the safe package. | No, intentionally excluded. |
| ROS2 CV bridge | No historical ROS2 CV bridge was found. | Missing. | `ros_bridge.py` optionally subscribes to camera images and publishes detections, tracks, debug images, and status only. | Yes, entirely new. |
| simulation demo | No safe historical demo media or reproducible video demo was recovered. | Missing. | Deterministic synthetic civilian video generation plus mock detector and IoU tracker pipeline config. | Yes, entirely new. |
| image inference CLI | No standalone safe historical CLI was found. | Missing. | `falcon infer-image` and legacy-compatible `falcon-vision-demo --input ...` run safe image inference. | Yes, entirely new. |
| offline fine-tuning | Notebook training-loop code and an interrupted attempt were found. | No completed historical run or checkpoint recovered. | Separate `falcon-train` CLI, audited group split, native-image tiles, binary public-weight initialization, checkpoints/resume, local CPU runs, and a prepared Colab notebook. | Yes, entirely new experiments; not historical recovery. |
| evaluation metrics | Notebook code referenced evaluation, but no completed historical metric output was recovered. | Incomplete. | New experiments use pycocotools on explicitly bounded positive-only validation subsets; results are recorded separately. No historical metrics are claimed. | Yes, new evaluation only; no metric reconstruction. |
