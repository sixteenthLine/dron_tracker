"""Aerial Object Detection; legacy import namespace retained for compatibility."""

from .coco_dataset import CocoDetectionDataset, DatasetValidationError, xywh_to_xyxy
from .detector import MODEL_NAME, DetectionResult, create_detector, predict_image
from .models import Detection, Frame, FrameResult, Track
from .pipeline import FrameProcessor, PipelineSummary
from .tracking import IoUTracker, LegacyGoturnUnavailable, MockTracker, NoOpTracker

__all__ = [
    "CocoDetectionDataset",
    "Detection",
    "DatasetValidationError",
    "DetectionResult",
    "Frame",
    "FrameProcessor",
    "FrameResult",
    "IoUTracker",
    "LegacyGoturnUnavailable",
    "MODEL_NAME",
    "MockTracker",
    "NoOpTracker",
    "PipelineSummary",
    "Track",
    "create_detector",
    "predict_image",
    "xywh_to_xyxy",
]
