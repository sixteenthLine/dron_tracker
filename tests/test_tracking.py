from PIL import Image

from falcon_vision_reconstructed.models import Detection, Frame
from falcon_vision_reconstructed.tracking import (
    IoUTracker,
    LegacyGoturnUnavailable,
    MockTracker,
    bbox_iou,
    create_tracker_backend,
)


def _frame(index: int = 0) -> Frame:
    return Frame(index=index, timestamp_sec=index / 12.0, image=Image.new("RGB", (160, 120)))


def _detection(box: tuple[float, float, float, float], label: str = "demo-object") -> Detection:
    return Detection(box=box, label=label, label_id=1, score=0.9)


def test_bbox_iou_for_overlapping_boxes() -> None:
    assert round(bbox_iou((0, 0, 20, 20), (10, 10, 30, 30)), 3) == 0.143


def test_iou_tracker_preserves_track_id_across_overlapping_updates() -> None:
    tracker = IoUTracker(iou_threshold=0.2)

    first_tracks = tracker.update([_detection((10, 10, 50, 40))], _frame(0))
    second_tracks = tracker.update([_detection((14, 12, 54, 42))], _frame(1))

    assert len(first_tracks) == 1
    assert len(second_tracks) == 1
    assert second_tracks[0].track_id == first_tracks[0].track_id
    assert second_tracks[0].age == 2
    assert second_tracks[0].missed == 0


def test_iou_tracker_creates_distinct_ids_for_multiple_objects() -> None:
    tracker = IoUTracker(iou_threshold=0.2)

    tracks = tracker.update(
        [
            _detection((10, 10, 40, 35), "first"),
            _detection((90, 20, 130, 60), "second"),
        ],
        _frame(0),
    )

    assert [track.track_id for track in tracks] == [1, 2]
    assert [track.label for track in tracks] == ["first", "second"]


def test_iou_tracker_drops_tracks_after_configured_misses() -> None:
    tracker = IoUTracker(iou_threshold=0.2, max_missed=1)

    tracker.update([_detection((10, 10, 50, 40))], _frame(0))
    still_alive = tracker.update([], _frame(1))
    removed = tracker.update([], _frame(2))

    assert len(still_alive) == 1
    assert still_alive[0].missed == 1
    assert removed == []


def test_iou_tracker_ignores_tiny_boxes() -> None:
    tracker = IoUTracker(min_box_area=100.0)

    tracks = tracker.update([_detection((10, 10, 12, 12))], _frame(0))

    assert tracks == []


def test_mock_tracker_returns_repeatable_test_tracks() -> None:
    tracker = MockTracker()

    tracks = tracker.update([_detection((10, 10, 40, 35))], _frame(2))

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].age == 3


def test_legacy_goturn_backend_is_unavailable_by_default() -> None:
    try:
        create_tracker_backend("legacy_goturn")
    except LegacyGoturnUnavailable as exc:
        assert "optional" in str(exc).lower()
    else:
        raise AssertionError("legacy GOTURN should not be enabled by default")
