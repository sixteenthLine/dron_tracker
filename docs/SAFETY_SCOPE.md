# Safety Scope

This reconstruction is a safe software-engineering portfolio demo for benign computer vision.

## Allowed Scope

- Public, civilian, synthetic, or otherwise non-sensitive images and videos.
- Still-image object detection.
- Offline fine-tuning on public civilian data, with explicitly new metrics.
- Synthetic video processing.
- Generic bounding boxes and class labels.
- Track IDs for generic object association.
- Debug overlays.
- Latency and FPS-style diagnostics.
- Optional ROS2 camera input and perception-output topics.

## Excluded Scope

- Real-drone integration.
- Simulated drone control.
- Navigation or movement.
- Autonomous following.
- Target selection or prioritization.
- Engagement, weapon, actuator, or trigger logic.
- Operational surveillance workflows.
- Identification of specific people.
- Training on sensitive datasets.
- Deployment instructions.

## Historical Boundary

The original historical folders are archived evidence. They are not modified by this reconstruction. Operationally sensitive fragments found during audit are documented only at a high level and are not reimplemented.

## Demo Media

Level B uses deterministic synthetic footage for video/tracking validation and
two attributed public aerial still images for detector smoke testing. Subsequent
offline training uses a separately audited civilian export. These assets and new
training results are not presented as historical project data or operational
readiness evidence.
