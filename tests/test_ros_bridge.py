import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from falcon_vision_reconstructed.models import Detection, Track
from falcon_vision_reconstructed.ros_bridge import (
    detections_payload,
    is_ros2_available,
    ros_image_to_pil,
    status_payload,
    topic_contract,
    tracks_payload,
)


def test_ros_topic_contract_is_output_only_except_camera_input() -> None:
    topics = topic_contract()

    assert topics == {
        "camera": "/camera/image_raw",
        "detections": "/falcon/detections",
        "tracks": "/falcon/tracks",
        "debug_image": "/falcon/debug_image",
        "status": "/falcon/status",
    }
    assert all("cmd" not in topic and "control" not in topic for topic in topics.values())


def test_ros_payloads_are_safe_json_measurements() -> None:
    detection = Detection(box=(1.0, 2.0, 11.0, 12.0), label="demo-object", label_id=1, score=0.95)
    track = Track(track_id=7, box=detection.box, label=detection.label, label_id=1, score=0.95, age=3, missed=0)

    detections = json.loads(detections_payload([detection], timestamp_sec=12.5))
    tracks = json.loads(tracks_payload([track], timestamp_sec=12.5))
    status = json.loads(status_payload(latency_ms=3.2, detections=1, tracks=1))

    assert detections["detections"][0]["bbox_xyxy"] == [1.0, 2.0, 11.0, 12.0]
    assert tracks["tracks"][0]["track_id"] == 7
    assert status["mode"] == "camera_detection_tracking_visualization_only"


def test_ros_image_to_pil_supports_rgb8_without_ros_runtime() -> None:
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    message = SimpleNamespace(
        encoding="rgb8",
        height=2,
        width=2,
        data=np.asarray(image).tobytes(),
    )

    converted = ros_image_to_pil(message)

    assert converted.size == (2, 2)
    assert converted.getpixel((0, 0)) == (10, 20, 30)


def test_ros_availability_check_returns_boolean() -> None:
    assert isinstance(is_ros2_available(), bool)
