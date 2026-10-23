# System Architecture

This document describes the Level B reconstruction and the subsequently added
offline training branch. It is not a historical claim of integration.

## Component Diagram

```mermaid
flowchart TD
    subgraph Sources
        A1[ImageFileSource]
        A2[VideoFileSource]
        A3[WebcamSource optional local read-only]
        A4[SyntheticFrameSource for tests/demo]
    end

    subgraph Core
        B[Detector protocol]
        B1[TorchvisionDetector]
        B2[MockDetector]
        C[Tracker protocol]
        C1[IoUTracker]
        C2[NoOpTracker]
        C3[MockTracker]
        C4[Legacy GOTURN unavailable by default]
        D[FrameProcessor]
    end

    subgraph Output
        E[draw_overlay]
        F[OutputVideoWriter]
        G[structured JSON logs]
    end

    A1 --> D
    A2 --> D
    A3 --> D
    A4 --> D
    D --> B
    B --> C
    C --> E
    E --> F
    D --> G
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Source as FrameSource
    participant Processor as FrameProcessor
    participant Detector as Detector
    participant Tracker as Tracker
    participant Overlay as Visualization
    participant Sink as FrameSink
    participant Logs as Structured logs

    Source->>Processor: next Frame
    Processor->>Detector: detect(image, frame_index)
    Detector-->>Processor: Detection[]
    Processor->>Tracker: update(detections, frame)
    Tracker-->>Processor: Track[]
    Processor->>Overlay: draw detections and tracks
    Overlay-->>Processor: annotated image
    Processor->>Sink: write(frame, annotated image)
    Processor->>Logs: frame index, counts, latency
```

## Data Model

- `Frame`: index, timestamp, and PIL RGB image.
- `Detection`: `xyxy` box, class label, class id, confidence.
- `Track`: stable track id, `xyxy` box, class label, confidence, age, missed count.
- `FrameResult`: frame plus detections, tracks, processed flag, and latency.

## Configuration

Pipeline configuration is loaded from TOML through `PipelineConfig`.

Supported knobs include:

- input and output paths;
- source type: image, video, webcam, or synthetic;
- detector backend: torchvision or mock;
- tracker backend: iou, none, mock, or legacy_goturn;
- confidence threshold;
- frame skipping;
- maximum frames;
- FPS;
- IoU threshold;
- track lifecycle;
- log level.

The safe reproducible demo uses `configs/synthetic_video_demo.toml`.

## Separate Offline Training Branch

```mermaid
flowchart TD
    A[Local civilian COCO export] --> B[Validate images, boxes, hashes]
    B --> C[Group split before tiling; exclude unannotated images]
    C --> D[Positive training tiles at native resolution]
    P[Public torchvision COCO weights] --> E[Binary Faster R-CNN initialization]
    D --> F[SGD fine-tuning]
    E --> F
    C --> G[Fixed full-frame validation grid]
    F --> G
    G --> H[pycocotools metrics; best/last checkpoints]
    H --> I[Separate offline image inference and overlay]
```

`training_data.py` owns preparation and crop translation; `training.py` owns
model initialization, training, resume, tiled inference, and COCO evaluation.
TOML configurations distinguish the bounded local pilot, local full training
split, and optional CUDA/Colab run. Checkpoints and data remain Git-ignored.
This branch does not replace the pretrained detector in the Level A/B pipeline
or connect training results to ROS2, tracking, or drone systems. See
[TRAINING.md](TRAINING.md).

## Safety Boundary

The architecture stops at visual perception outputs: detections, tracks, overlays, and diagnostics. It has no interface for navigation, target selection, actuation, mission planning, or command publication.
