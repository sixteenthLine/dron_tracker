"""Optional ROS2 bridge for camera-only CV integration.

This module is new reconstruction work. It publishes detections, tracks, debug
images, and status diagnostics only. It never publishes motion, navigation,
actuator, mission, or control commands.
"""

from __future__ import annotations

import json
import time
from typing import Any

from PIL import Image
import numpy as np

from .detector import create_detector_backend
from .models import Detection, Frame, Track
from .tracking import create_tracker_backend
from .visualization import draw_overlay


TOPICS = {
    "camera": "/camera/image_raw",
    "detections": "/falcon/detections",
    "tracks": "/falcon/tracks",
    "debug_image": "/falcon/debug_image",
    "status": "/falcon/status",
}


def is_ros2_available() -> bool:
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
        import std_msgs.msg  # noqa: F401
    except ImportError:
        return False
    return True


def topic_contract() -> dict[str, str]:
    return dict(TOPICS)


def detections_payload(detections: list[Detection], *, timestamp_sec: float) -> str:
    return json.dumps(
        {
            "timestamp_sec": timestamp_sec,
            "detections": [
                {
                    "bbox_xyxy": list(detection.box),
                    "class_label": detection.label,
                    "class_id": detection.label_id,
                    "confidence": detection.score,
                }
                for detection in detections
            ],
        },
        sort_keys=True,
    )


def tracks_payload(tracks: list[Track], *, timestamp_sec: float) -> str:
    return json.dumps(
        {
            "timestamp_sec": timestamp_sec,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox_xyxy": list(track.box),
                    "class_label": track.label,
                    "class_id": track.label_id,
                    "confidence": track.score,
                    "age": track.age,
                    "missed": track.missed,
                }
                for track in tracks
            ],
        },
        sort_keys=True,
    )


def status_payload(*, latency_ms: float, detections: int, tracks: int) -> str:
    return json.dumps(
        {
            "node": "falcon_cv_bridge",
            "latency_ms": latency_ms,
            "detections": detections,
            "tracks": tracks,
            "mode": "camera_detection_tracking_visualization_only",
        },
        sort_keys=True,
    )


def ros_image_to_pil(message: Any) -> Image.Image:
    encoding = getattr(message, "encoding", "")
    height = int(getattr(message, "height"))
    width = int(getattr(message, "width"))
    data = bytes(getattr(message, "data"))
    if encoding in {"rgb8", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        return Image.fromarray(array).convert("RGB")
    if encoding == "mono8":
        array = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        return Image.fromarray(array).convert("RGB")
    raise ValueError(f"Unsupported ROS image encoding: {encoding}")


def pil_to_ros_image(image: Image.Image, *, header: Any | None = None) -> Any:
    from sensor_msgs.msg import Image as RosImage

    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    message = RosImage()
    if header is not None:
        message.header = header
    message.height = array.shape[0]
    message.width = array.shape[1]
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = array.shape[1] * 3
    message.data = array.tobytes()
    return message


def build_node_class() -> type[Any]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import String

    class FalconCvBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("falcon_cv_bridge")
            self.detector = create_detector_backend("mock", threshold=0.5)
            self.tracker = create_tracker_backend("iou")
            self.frame_index = 0
            self.detections_pub = self.create_publisher(String, TOPICS["detections"], 10)
            self.tracks_pub = self.create_publisher(String, TOPICS["tracks"], 10)
            self.status_pub = self.create_publisher(String, TOPICS["status"], 10)
            self.debug_image_pub = self.create_publisher(RosImage, TOPICS["debug_image"], 10)
            self.subscription = self.create_subscription(RosImage, TOPICS["camera"], self._on_image, 10)

        def _on_image(self, message: Any) -> None:
            start = time.perf_counter()
            pil_image = ros_image_to_pil(message)
            frame = Frame(index=self.frame_index, timestamp_sec=time.time(), image=pil_image)
            detections = self.detector.detect(pil_image, frame_index=self.frame_index)
            tracks = self.tracker.update(detections, frame)
            overlay = draw_overlay(pil_image, detections=detections, tracks=tracks)
            latency_ms = (time.perf_counter() - start) * 1000.0

            self.detections_pub.publish(String(data=detections_payload(detections, timestamp_sec=frame.timestamp_sec)))
            self.tracks_pub.publish(String(data=tracks_payload(tracks, timestamp_sec=frame.timestamp_sec)))
            self.status_pub.publish(
                String(data=status_payload(latency_ms=latency_ms, detections=len(detections), tracks=len(tracks)))
            )
            self.debug_image_pub.publish(pil_to_ros_image(overlay, header=getattr(message, "header", None)))
            self.frame_index += 1

    return FalconCvBridgeNode


def run_ros_node() -> int:
    if not is_ros2_available():
        print("ROS2 is not installed; core package remains usable without ROS.")
        return 1

    import rclpy

    node_class = build_node_class()
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0

