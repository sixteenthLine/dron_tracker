# Tracking Backends

Level B defines a common `Tracker` protocol so tracking backends can be swapped without changing the frame processor.

## IoUTracker

`IoUTracker` is the safe default modern tracker backend.

It provides:

- stable track IDs across overlapping detections;
- configurable IoU association threshold;
- configurable `max_missed` lifecycle;
- configurable minimum box area;
- no identity recognition;
- no face recognition;
- no control, navigation, or target-prioritization output.

It is suitable for deterministic synthetic demos and small civilian portfolio examples. It is not presented as a production multi-object tracking benchmark.

## NoOpTracker

`NoOpTracker` exposes detections as one-frame tracks. It is useful when the user wants detection overlays without temporal association.

## MockTracker

`MockTracker` is a deterministic test backend. It assigns repeatable IDs by detection order and exists for unit/integration tests, not for model-quality claims.

## Legacy GOTURN Boundary

Historical evidence showed an OpenCV/GOTURN tracking experiment in a separate branch. The current reconstruction does not copy, modify, redistribute, or silently enable those historical files.

The `legacy_goturn` backend is intentionally unavailable by default because:

- GOTURN model-file license and provenance need review before redistribution;
- historical code used old OpenCV/contrib APIs;
- compatibility depends on local model/prototxt paths;
- the safe Level B demo does not need GOTURN.

`LegacyGoturnTracker` exists only as a compatibility boundary for future local experiments after dependency and provenance review. The supported safe demo uses `IoUTracker`.

Installing the optional OpenCV extra or supplying local model paths does not
make the current wrapper functional: construction always raises
`LegacyGoturnUnavailable`. No working GOTURN backend is advertised.
