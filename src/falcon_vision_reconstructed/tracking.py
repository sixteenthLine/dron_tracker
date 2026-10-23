"""Replaceable tracker backends for the safe Level B prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import BBox, Detection, Frame, Track


def bbox_iou(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    box: BBox
    label: str
    label_id: int
    score: float
    age: int = 1
    missed: int = 0

    def as_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            box=self.box,
            label=self.label,
            label_id=self.label_id,
            score=self.score,
            age=self.age,
            missed=self.missed,
        )


class NoOpTracker:
    """Tracker backend that exposes detections as one-frame tracks."""

    name = "none"

    def __init__(self) -> None:
        self._tracks: list[Track] = []

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        del frame
        self._tracks = [
            Track(
                track_id=index + 1,
                box=detection.box,
                label=detection.label,
                label_id=detection.label_id,
                score=detection.score,
                age=1,
                missed=0,
            )
            for index, detection in enumerate(detections)
        ]
        return self.current_tracks()

    def current_tracks(self) -> list[Track]:
        return list(self._tracks)


class MockTracker:
    """Deterministic tracker for tests.

    It assigns repeatable IDs by detection order and does not infer identity
    beyond the current synthetic/test sequence.
    """

    name = "mock-tracker"

    def __init__(self) -> None:
        self._tracks: list[Track] = []

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        self._tracks = [
            Track(
                track_id=index + 1,
                box=detection.box,
                label=detection.label,
                label_id=detection.label_id,
                score=detection.score,
                age=frame.index + 1,
                missed=0,
            )
            for index, detection in enumerate(detections)
        ]
        return self.current_tracks()

    def current_tracks(self) -> list[Track]:
        return list(self._tracks)


class IoUTracker:
    """Simple modern generic tracker based on IoU association.

    This tracker assigns stable object IDs across frames based on bounding-box
    overlap. It does not perform identity recognition, face recognition,
    navigation, or control output.
    """

    name = "iou-tracker"

    def __init__(self, *, iou_threshold: float = 0.3, max_missed: int = 5, min_box_area: float = 25.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_box_area = min_box_area
        self._next_id = 1
        self._states: list[_TrackState] = []

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        del frame
        valid_detections = [detection for detection in detections if self._area(detection.box) >= self.min_box_area]
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for detection_index, detection in enumerate(valid_detections):
            best_state_index: int | None = None
            best_iou = 0.0
            for state_index, state in enumerate(self._states):
                if state_index in matched_tracks:
                    continue
                score = bbox_iou(state.box, detection.box)
                if score > best_iou:
                    best_iou = score
                    best_state_index = state_index

            if best_state_index is not None and best_iou >= self.iou_threshold:
                state = self._states[best_state_index]
                state.box = detection.box
                state.label = detection.label
                state.label_id = detection.label_id
                state.score = detection.score
                state.age += 1
                state.missed = 0
                matched_tracks.add(best_state_index)
                matched_detections.add(detection_index)

        for state_index, state in enumerate(self._states):
            if state_index not in matched_tracks:
                state.age += 1
                state.missed += 1

        for detection_index, detection in enumerate(valid_detections):
            if detection_index in matched_detections:
                continue
            self._states.append(
                _TrackState(
                    track_id=self._next_id,
                    box=detection.box,
                    label=detection.label,
                    label_id=detection.label_id,
                    score=detection.score,
                )
            )
            self._next_id += 1

        self._states = [state for state in self._states if state.missed <= self.max_missed]
        return self.current_tracks()

    def current_tracks(self) -> list[Track]:
        return [state.as_track() for state in self._states]

    @staticmethod
    def _area(box: BBox) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class LegacyGoturnUnavailable(RuntimeError):
    """Raised when the optional reconstructed legacy GOTURN wrapper cannot run."""


class LegacyGoturnTracker:
    """Compatibility stub for the historical OpenCV GOTURN branch.

    The archived GOTURN model files are not redistributed here. This wrapper is
    intentionally optional and only documents the boundary for future local
    experiments after license/provenance review.
    """

    name = "legacy-goturn-unavailable"

    def __init__(self, *, model_path: str | Path | None = None, prototxt_path: str | Path | None = None) -> None:
        if model_path is None or prototxt_path is None:
            raise LegacyGoturnUnavailable(
                "Legacy GOTURN requires local model_path and prototxt_path after provenance review."
            )
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise LegacyGoturnUnavailable("Legacy GOTURN requires optional OpenCV contrib dependencies.") from exc
        raise LegacyGoturnUnavailable(
            "Legacy GOTURN compatibility is documented but not enabled in the safe Level B demo."
        )

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        del detections, frame
        raise LegacyGoturnUnavailable("Legacy GOTURN is not enabled.")

    def current_tracks(self) -> list[Track]:
        return []


def create_tracker_backend(
    backend: str,
    *,
    iou_threshold: float = 0.3,
    max_missed: int = 5,
    min_box_area: float = 25.0,
) -> IoUTracker | NoOpTracker | MockTracker:
    if backend == "iou":
        return IoUTracker(iou_threshold=iou_threshold, max_missed=max_missed, min_box_area=min_box_area)
    if backend == "none":
        return NoOpTracker()
    if backend == "mock":
        return MockTracker()
    if backend == "legacy_goturn":
        raise LegacyGoturnUnavailable(
            "Legacy GOTURN is optional and unavailable by default; use tracker_backend='iou' for the safe demo."
        )
    raise ValueError(f"Unsupported tracker backend: {backend}")
