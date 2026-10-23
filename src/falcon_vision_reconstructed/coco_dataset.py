"""COCO-style dataset utilities for the safe reconstruction.

This module is inspired by the preserved historical ``Dataprep.py`` idea:
read COCO-style annotations, convert boxes, and return torchvision-style
detection targets. The implementation here is new reconstruction code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class DatasetValidationError(ValueError):
    """Raised when a COCO annotation file or referenced image is invalid."""


Target = dict[str, Tensor]
Transform = Callable[[Image.Image, Target], tuple[Any, Target]]


@dataclass(frozen=True)
class _ImageRecord:
    image_id: int
    file_name: str


@dataclass(frozen=True)
class _AnnotationRecord:
    box_xyxy: tuple[float, float, float, float]
    label: int
    area: float
    iscrowd: int


def xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a COCO ``[x, y, width, height]`` box to ``[x1, y1, x2, y2]``."""

    if len(bbox) != 4:
        raise DatasetValidationError("COCO bbox must contain exactly 4 values.")

    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError("COCO bbox values must be numeric.") from exc

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise DatasetValidationError("COCO bbox values must be finite numbers.")
    if width <= 0 or height <= 0:
        raise DatasetValidationError("COCO bbox width and height must be positive.")

    return x, y, x + width, y + height


class CocoDetectionDataset(Dataset):
    """A small validated COCO-style object-detection dataset.

    Labels default to contiguous positive integers sorted by COCO category id.
    Pass ``category_id_to_label`` to keep a custom label scheme.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        *,
        category_id_to_label: Mapping[int, int] | None = None,
        include_category_ids: Iterable[int] | None = None,
        transforms: Transform | None = None,
        validate_images: bool = False,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.transforms = transforms
        self._include_category_ids = (
            set(int(category_id) for category_id in include_category_ids)
            if include_category_ids is not None
            else None
        )

        payload = self._load_json(self.annotation_file)
        images, categories, annotations = self._read_top_level_sections(payload)

        self._category_names = self._parse_categories(categories)
        self._category_id_to_label = self._build_label_map(category_id_to_label)
        self.label_names = self._build_label_names()
        self._images = self._parse_images(images)
        self._annotations_by_image = self._parse_annotations(annotations)

        if validate_images:
            self._validate_all_image_files()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Target]:
        record = self._images[index]
        image_path = self.image_root / record.file_name
        if not image_path.is_file():
            raise DatasetValidationError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        annotations = self._annotations_by_image.get(record.image_id, [])
        target = self._target_from_annotations(record.image_id, annotations)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def categories(self) -> Mapping[int, str]:
        """COCO category id to category name."""

        return dict(self._category_names)

    @staticmethod
    def _load_json(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            raise DatasetValidationError(f"Annotation file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Annotation file is not valid JSON: {path}") from exc

        if not isinstance(payload, dict):
            raise DatasetValidationError("COCO annotation root must be a JSON object.")
        return payload

    @staticmethod
    def _read_top_level_sections(payload: Mapping[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
        missing = [key for key in ("images", "annotations", "categories") if key not in payload]
        if missing:
            raise DatasetValidationError(f"COCO annotation file is missing sections: {', '.join(missing)}")

        images = payload["images"]
        annotations = payload["annotations"]
        categories = payload["categories"]

        if not isinstance(images, list):
            raise DatasetValidationError("COCO 'images' section must be a list.")
        if not isinstance(annotations, list):
            raise DatasetValidationError("COCO 'annotations' section must be a list.")
        if not isinstance(categories, list):
            raise DatasetValidationError("COCO 'categories' section must be a list.")

        return images, categories, annotations

    def _parse_categories(self, categories: list[Any]) -> dict[int, str]:
        category_names: dict[int, str] = {}
        for raw_category in categories:
            if not isinstance(raw_category, dict):
                raise DatasetValidationError("Each COCO category must be an object.")
            category_id = self._required_int(raw_category, "id", "category")
            name = raw_category.get("name")
            if not isinstance(name, str) or not name.strip():
                raise DatasetValidationError(f"Category {category_id} must have a non-empty string name.")
            if category_id in category_names:
                raise DatasetValidationError(f"Duplicate category id: {category_id}")
            category_names[category_id] = name

        if not category_names:
            raise DatasetValidationError("At least one COCO category is required.")
        return category_names

    def _build_label_map(self, category_id_to_label: Mapping[int, int] | None) -> dict[int, int]:
        if category_id_to_label is None:
            included_ids = self._included_category_ids()
            return {category_id: label for label, category_id in enumerate(sorted(included_ids), start=1)}

        label_map: dict[int, int] = {}
        for raw_category_id, raw_label in category_id_to_label.items():
            category_id = int(raw_category_id)
            label = int(raw_label)
            if category_id not in self._category_names:
                raise DatasetValidationError(f"Label map references unknown category id: {category_id}")
            if label <= 0:
                raise DatasetValidationError("Detection labels must be positive integers.")
            label_map[category_id] = label

        missing_ids = self._included_category_ids() - set(label_map)
        if missing_ids:
            formatted = ", ".join(str(category_id) for category_id in sorted(missing_ids))
            raise DatasetValidationError(f"Label map is missing included category ids: {formatted}")
        return label_map

    def _included_category_ids(self) -> set[int]:
        if self._include_category_ids is None:
            return set(self._category_names)

        unknown = self._include_category_ids - set(self._category_names)
        if unknown:
            formatted = ", ".join(str(category_id) for category_id in sorted(unknown))
            raise DatasetValidationError(f"Included category ids are unknown: {formatted}")
        return set(self._include_category_ids)

    def _build_label_names(self) -> dict[int, str]:
        label_names: dict[int, str] = {}
        for category_id, label in self._category_id_to_label.items():
            label_names.setdefault(label, self._category_names[category_id])
        return label_names

    def _parse_images(self, images: list[Any]) -> list[_ImageRecord]:
        records: list[_ImageRecord] = []
        seen_ids: set[int] = set()
        for raw_image in images:
            if not isinstance(raw_image, dict):
                raise DatasetValidationError("Each COCO image entry must be an object.")
            image_id = self._required_int(raw_image, "id", "image")
            file_name = raw_image.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                raise DatasetValidationError(f"Image {image_id} must have a non-empty file_name.")
            if image_id in seen_ids:
                raise DatasetValidationError(f"Duplicate image id: {image_id}")
            seen_ids.add(image_id)
            records.append(_ImageRecord(image_id=image_id, file_name=file_name))

        if not records:
            raise DatasetValidationError("COCO 'images' section must contain at least one image.")
        return records

    def _parse_annotations(self, annotations: list[Any]) -> dict[int, list[_AnnotationRecord]]:
        image_ids = {record.image_id for record in self._images}
        annotations_by_image: dict[int, list[_AnnotationRecord]] = defaultdict(list)
        included_ids = self._included_category_ids()

        for raw_annotation in annotations:
            if not isinstance(raw_annotation, dict):
                raise DatasetValidationError("Each COCO annotation must be an object.")

            image_id = self._required_int(raw_annotation, "image_id", "annotation")
            if image_id not in image_ids:
                raise DatasetValidationError(f"Annotation references unknown image id: {image_id}")

            category_id = self._required_int(raw_annotation, "category_id", "annotation")
            if category_id not in self._category_names:
                raise DatasetValidationError(f"Annotation references unknown category id: {category_id}")
            if category_id not in included_ids:
                continue

            bbox = raw_annotation.get("bbox")
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
                raise DatasetValidationError("COCO annotation bbox must be a sequence of 4 numbers.")

            box_xyxy = xywh_to_xyxy(bbox)
            width = box_xyxy[2] - box_xyxy[0]
            height = box_xyxy[3] - box_xyxy[1]
            iscrowd = int(raw_annotation.get("iscrowd", 0))
            annotations_by_image[image_id].append(
                _AnnotationRecord(
                    box_xyxy=box_xyxy,
                    label=self._category_id_to_label[category_id],
                    area=width * height,
                    iscrowd=iscrowd,
                )
            )

        return dict(annotations_by_image)

    def _target_from_annotations(self, image_id: int, annotations: list[_AnnotationRecord]) -> Target:
        if not annotations:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor([ann.box_xyxy for ann in annotations], dtype=torch.float32),
            "labels": torch.tensor([ann.label for ann in annotations], dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor([ann.area for ann in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor([ann.iscrowd for ann in annotations], dtype=torch.int64),
        }

    def _validate_all_image_files(self) -> None:
        missing = [record.file_name for record in self._images if not (self.image_root / record.file_name).is_file()]
        if missing:
            preview = ", ".join(missing[:3])
            if len(missing) > 3:
                preview += f", and {len(missing) - 3} more"
            raise DatasetValidationError(f"Missing image files: {preview}")

    @staticmethod
    def _required_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DatasetValidationError(f"COCO {context} field '{key}' must be an integer.")
        return value
