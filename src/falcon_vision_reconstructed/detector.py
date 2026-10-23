"""Detector implementations for the safe reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import to_tensor

from .models import Detection


MODEL_NAME = "torchvision Faster R-CNN MobileNet V3 Large 320 FPN, COCO pretrained weights"


@dataclass(frozen=True)
class DetectionResult:
    boxes: list[tuple[float, float, float, float]]
    labels: list[str]
    label_ids: list[int]
    scores: list[float]

    def as_detections(self) -> list[Detection]:
        return [
            Detection(box=box, label=label, label_id=label_id, score=score)
            for box, label, label_id, score in zip(self.boxes, self.labels, self.label_ids, self.scores)
        ]


def create_detector(device: str = "cpu") -> tuple[torch.nn.Module, list[str]]:
    """Create a modern pretrained torchvision detector in eval mode."""

    resolved_device = torch.device(device)
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.to(resolved_device)
    model.eval()
    categories = list(weights.meta.get("categories", []))
    return model, categories


def predict_image(
    image: Image.Image,
    *,
    threshold: float = 0.5,
    max_detections: int = 20,
    device: str = "cpu",
    model: torch.nn.Module | None = None,
    categories: list[str] | None = None,
) -> DetectionResult:
    """Run detector inference on one PIL image and return filtered predictions."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    if model is None or categories is None:
        model, categories = create_detector(device=device)

    resolved_device = torch.device(device)
    rgb_image = image.convert("RGB")
    image_tensor = to_tensor(rgb_image).to(resolved_device)

    with torch.inference_mode():
        raw_output: dict[str, Any] = model([image_tensor])[0]

    boxes_tensor = raw_output["boxes"].detach().cpu()
    labels_tensor = raw_output["labels"].detach().cpu()
    scores_tensor = raw_output["scores"].detach().cpu()

    selected_boxes: list[tuple[float, float, float, float]] = []
    selected_label_ids: list[int] = []
    selected_labels: list[str] = []
    selected_scores: list[float] = []

    for box, label_id_tensor, score_tensor in zip(boxes_tensor, labels_tensor, scores_tensor):
        score = float(score_tensor.item())
        if score < threshold:
            continue
        label_id = int(label_id_tensor.item())
        label = categories[label_id] if 0 <= label_id < len(categories) else f"class_{label_id}"
        selected_boxes.append(tuple(float(value) for value in box.tolist()))
        selected_label_ids.append(label_id)
        selected_labels.append(label)
        selected_scores.append(score)
        if len(selected_boxes) >= max_detections:
            break

    return DetectionResult(
        boxes=selected_boxes,
        labels=selected_labels,
        label_ids=selected_label_ids,
        scores=selected_scores,
    )


class TorchvisionDetector:
    """Detector interface wrapper around public torchvision pretrained weights."""

    name = "torchvision-fasterrcnn-mobilenet-v3-320-fpn"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        max_detections: int = 20,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.max_detections = max_detections
        self.device = device
        self._model, self._categories = create_detector(device=device)

    def detect(self, image: Image.Image, *, frame_index: int = 0) -> list[Detection]:
        del frame_index
        result = predict_image(
            image,
            threshold=self.threshold,
            max_detections=self.max_detections,
            device=self.device,
            model=self._model,
            categories=self._categories,
        )
        return result.as_detections()


class MockDetector:
    """Deterministic detector for tests and synthetic demos.

    This does not perform model inference. It returns a predictable moving box
    so pipeline and tracking behavior can be validated without network access,
    heavyweight model execution, or sensitive media.
    """

    name = "mock-detector"

    def __init__(self, *, threshold: float = 0.5, label: str = "demo-object") -> None:
        self.threshold = threshold
        self.label = label

    def detect(self, image: Image.Image, *, frame_index: int = 0) -> list[Detection]:
        if self.threshold > 0.95:
            return []
        width, height = image.size
        x1 = min(width - 42, 38 + frame_index * 5)
        y1 = max(10, int(height * 0.55))
        return [
            Detection(
                box=(float(x1), float(y1), float(x1 + 52), float(y1 + 30)),
                label=self.label,
                label_id=1,
                score=0.99,
            )
        ]


def create_detector_backend(
    backend: str,
    *,
    threshold: float = 0.5,
    max_detections: int = 20,
    device: str = "cpu",
) -> TorchvisionDetector | MockDetector:
    if backend == "torchvision":
        return TorchvisionDetector(threshold=threshold, max_detections=max_detections, device=device)
    if backend == "mock":
        return MockDetector(threshold=threshold)
    raise ValueError(f"Unsupported detector backend: {backend}")
