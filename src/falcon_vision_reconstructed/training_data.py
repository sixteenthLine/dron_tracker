"""Audited, group-separated data for new offline civilian training experiments."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

from .coco_dataset import CocoDetectionDataset, xywh_to_xyxy


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_group(file_name: str) -> str:
    parts = Path(file_name).name.split("_")
    if len(parts) < 3 or parts[0] != "train" or not parts[1]:
        raise ValueError(f"Cannot infer a scene group from file name: {file_name}")
    return parts[1]


def grid_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= 0 or size <= 0 or not 0 <= overlap < size:
        raise ValueError("Tile dimensions must be positive and 0 <= overlap < tile size")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def clipped_boxes(boxes: list[list[float]], crop: tuple[int, int, int, int]) -> list[list[float]]:
    x1, y1, x2, y2 = crop
    result = []
    for left, top, right, bottom in boxes:
        a, b, c, d = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
        if c > a and d > b and (c - a) * (d - b) >= 0.5 * (right - left) * (bottom - top):
            result.append([a - x1, b - y1, c - x1, d - y1])
    return result


def prepare_data(
    dataset: Path,
    output: Path,
    *,
    val_groups: tuple[str, ...] = ("BRK", "RAK"),
    test_groups: tuple[str, ...] = ("VRD", "ZRI"),
) -> dict[str, Any]:
    if set(val_groups) & set(test_groups):
        raise ValueError("Validation and test groups must not overlap")
    annotation_path = dataset / "_annotations.coco.json"
    image_root = dataset / "train"
    CocoDetectionDataset(annotation_path, image_root, include_category_ids=[1], category_id_to_label={1: 1})
    data = json.loads(annotation_path.read_text())
    annotations: dict[int, list[list[float]]] = {}
    image_meta = {record["id"]: record for record in data["images"]}
    for annotation in data["annotations"]:
        if annotation["category_id"] != 1 or annotation.get("iscrowd", 0):
            raise ValueError("This training experiment requires non-crowd category_id=1 annotations")
        box = list(xywh_to_xyxy(annotation["bbox"]))
        record = image_meta[annotation["image_id"]]
        if box[0] < 0 or box[1] < 0 or box[2] > record["width"] or box[3] > record["height"]:
            raise ValueError(f"Out-of-bounds box in image {record['id']}")
        annotations.setdefault(record["id"], []).append(box)

    known_groups = {scene_group(record["file_name"]) for record in data["images"]}
    if not set(val_groups + test_groups) <= known_groups:
        raise ValueError("Requested validation/test groups are absent from the export")
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    excluded: list[int] = []
    fingerprints: dict[str, tuple[str, str]] = {}
    duplicates = 0
    for index, record in enumerate(data["images"]):
        file_name = record["file_name"]
        if Path(file_name).name != file_name:
            raise ValueError("Image file_name must be a basename within the train directory")
        path = image_root / file_name
        if not path.is_file():
            raise ValueError(f"Missing image: {file_name}")
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Image dimensions disagree with annotations: {file_name}")
            image.verify()
        fingerprint = file_sha256(path)
        group = scene_group(file_name)
        split = "test" if group in test_groups else "val" if group in val_groups else "train"
        if fingerprint in fingerprints:
            old_split, _ = fingerprints[fingerprint]
            if old_split != split:
                raise ValueError("An exact duplicate image crosses group splits; resolve before training")
            duplicates += 1
            continue
        fingerprints[fingerprint] = (split, file_name)
        if not annotations.get(record["id"]):
            excluded.append(record["id"])
            continue
        splits[split].append({
            "id": record["id"], "file_name": file_name, "group": group,
            "width": record["width"], "height": record["height"],
            "sha256": fingerprint, "boxes": annotations[record["id"]],
        })
        if index % 250 == 0:
            print(json.dumps({"event": "audit_progress", "images_checked": index + 1}), flush=True)
    if any(not records for records in splits.values()):
        raise ValueError("Every split needs at least one annotated image")
    manifest = {
        "format_version": 1, "annotation_sha256": file_sha256(annotation_path),
        "category_id_to_label": {"1": 1}, "labels": ["background", "human"],
        "grouping": "filename prefix is a scene proxy, not verified geographic independence",
        "validation_groups": list(val_groups), "test_groups": list(test_groups),
        "excluded_unannotated_ids": excluded, "splits": splits,
    }
    audit = {
        "images_checked": len(data["images"]), "annotations_checked": len(data["annotations"]),
        "missing_files": 0, "invalid_boxes": 0, "exact_duplicates_removed": duplicates,
        "unannotated_excluded": len(excluded),
        "annotation_completeness": "unverified; empty images are not assumed to be background",
        "positive_only_bias": "training and reported validation exclude unverified empty images",
        "splits": {key: {"images": len(records), "groups": dict(Counter(r["group"] for r in records)),
                         "boxes": sum(len(r["boxes"]) for r in records)} for key, records in splits.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


class TrainingTiles(torch.utils.data.Dataset):
    """Positive grid tiles; no full-size image is downsampled before cropping."""

    def __init__(self, records: list[dict[str, Any]], image_root: Path, *, size: int, overlap: int) -> None:
        self.records = records
        self.image_root = image_root
        self.tiles: list[tuple[int, tuple[int, int, int, int], list[list[float]]]] = []
        for index, record in enumerate(records):
            for top in grid_starts(record["height"], size, overlap):
                for left in grid_starts(record["width"], size, overlap):
                    crop = (left, top, min(left + size, record["width"]), min(top + size, record["height"]))
                    boxes = clipped_boxes(record["boxes"], crop)
                    if boxes:
                        self.tiles.append((index, crop, boxes))
        if not self.tiles:
            raise ValueError("No positive training tiles were produced")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record_index, crop, boxes = self.tiles[index]
        record = self.records[record_index]
        with Image.open(self.image_root / record["file_name"]) as image:
            tensor = to_tensor(image.convert("RGB").crop(crop))
        return tensor, {"boxes": torch.tensor(boxes, dtype=torch.float32),
                        "labels": torch.ones(len(boxes), dtype=torch.int64),
                        "image_id": torch.tensor([record["id"]])}
