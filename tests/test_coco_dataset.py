import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from falcon_vision_reconstructed import CocoDetectionDataset, DatasetValidationError


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    Image.new("RGB", size, color=(230, 238, 240)).save(path)


def _write_coco(
    path: Path,
    *,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> Path:
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories or [{"id": 10, "name": "civilian-object"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_one_image_with_one_box_and_configurable_label(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "one.png")
    annotation_file = _write_coco(
        tmp_path / "annotations.json",
        images=[{"id": 1, "file_name": "one.png"}],
        annotations=[{"image_id": 1, "category_id": 10, "bbox": [5, 6, 10, 20]}],
    )

    dataset = CocoDetectionDataset(
        annotation_file,
        image_root,
        category_id_to_label={10: 7},
    )
    image, target = dataset[0]

    assert image.size == (64, 48)
    torch.testing.assert_close(target["boxes"], torch.tensor([[5.0, 6.0, 15.0, 26.0]]))
    assert target["labels"].tolist() == [7]
    assert dataset.label_names == {7: "civilian-object"}


def test_one_image_with_multiple_boxes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "many.png")
    annotation_file = _write_coco(
        tmp_path / "annotations.json",
        images=[{"id": 2, "file_name": "many.png"}],
        categories=[
            {"id": 10, "name": "civilian-object"},
            {"id": 20, "name": "vehicle"},
        ],
        annotations=[
            {"image_id": 2, "category_id": 10, "bbox": [1, 2, 3, 4]},
            {"image_id": 2, "category_id": 20, "bbox": [10, 12, 8, 9]},
        ],
    )

    dataset = CocoDetectionDataset(annotation_file, image_root)
    _, target = dataset[0]

    torch.testing.assert_close(
        target["boxes"],
        torch.tensor([[1.0, 2.0, 4.0, 6.0], [10.0, 12.0, 18.0, 21.0]]),
    )
    assert target["labels"].tolist() == [1, 2]
    torch.testing.assert_close(target["area"], torch.tensor([12.0, 72.0]))


def test_one_image_with_no_boxes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "empty.png")
    annotation_file = _write_coco(
        tmp_path / "annotations.json",
        images=[{"id": 3, "file_name": "empty.png"}],
        annotations=[],
    )

    dataset = CocoDetectionDataset(annotation_file, image_root)
    _, target = dataset[0]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["area"].shape == (0,)
    assert target["iscrowd"].shape == (0,)
    assert target["image_id"].tolist() == [3]


def test_missing_image_raises_clear_validation_error(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    annotation_file = _write_coco(
        tmp_path / "annotations.json",
        images=[{"id": 4, "file_name": "missing.png"}],
        annotations=[],
    )

    dataset = CocoDetectionDataset(annotation_file, image_root)

    with pytest.raises(DatasetValidationError, match="Image file not found"):
        dataset[0]


def test_malformed_annotation_raises_clear_validation_error(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "bad.png")
    annotation_file = _write_coco(
        tmp_path / "annotations.json",
        images=[{"id": 5, "file_name": "bad.png"}],
        annotations=[{"image_id": 5, "category_id": 10, "bbox": [1, 2, 3]}],
    )

    with pytest.raises(DatasetValidationError, match="exactly 4 values"):
        CocoDetectionDataset(annotation_file, image_root)

