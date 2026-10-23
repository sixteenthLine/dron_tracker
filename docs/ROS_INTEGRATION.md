# ROS2 Integration

The ROS2 bridge is new Level B reconstruction work. It is optional and isolated so the core package runs without ROS installed.

Current evidence: availability checks and pure helper tests passed; an actual
ROS2 graph/node was not started because the local runtime is unavailable. The
node's detector is currently fixed to the mock backend. A successful
`--check-only` exit is not proof that the node runs or delivers topics.

## Scope

The bridge accepts camera images, runs the detector/tracker pipeline, and publishes CV outputs and diagnostics. It does not publish control commands or connect to drone movement.

## Topic Contract

| direction | topic | payload |
|---|---|---|
| subscribe | `/camera/image_raw` | `sensor_msgs/Image` |
| publish | `/falcon/detections` | `std_msgs/String` containing JSON detections |
| publish | `/falcon/tracks` | `std_msgs/String` containing JSON tracks |
| publish | `/falcon/debug_image` | `sensor_msgs/Image` with overlays |
| publish | `/falcon/status` | `std_msgs/String` containing JSON diagnostics |

The JSON payloads are limited to bounding boxes, class labels, confidences, track IDs, timestamps, and latency.

## Explicit Non-Goals

The bridge does not publish:

- velocity commands;
- navigation goals;
- actuator commands;
- mission commands;
- target-selection commands;
- engagement or trigger commands.

## Local Check

```bash
aerial-detect ros-node --check-only
```

This prints ROS2 availability and the safe topic contract.

## Runtime

When ROS2 Python packages are installed and sourced in the shell:

```bash
aerial-detect ros-node
```

If ROS2 is unavailable, the command exits cleanly and the rest of the package remains usable.

## Testing Strategy

Unit tests validate:

- topic names;
- absence of command/control topic names;
- JSON payload shape;
- ROS image conversion helpers that do not require an active ROS graph.
