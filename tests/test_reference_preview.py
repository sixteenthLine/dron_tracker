import json
from pathlib import Path

from PIL import Image
import pytest

from falcon_vision_reconstructed.coco_dataset import CocoDetectionDataset
from scripts.render_reference_previews import render_preview, render_previews


ROOT = Path(__file__).resolve().parents[1]


def test_reference_annotations_are_scaled_source_boxes_not_predictions() -> None:
    data = json.loads((ROOT / "assets/aerial/reference_annotations.coco.json").read_text())
    assert "NOT model predictions" in data["info"]["description"]
    assert len(data["images"]) == 2 and len(data["annotations"]) == 16
    assert data["annotations"][0]["bbox"] == pytest.approx([571.2, 601.6, 23.2, 25.2])
    assert all("score" not in annotation for annotation in data["annotations"])
    dataset = CocoDetectionDataset(ROOT / "assets/aerial/reference_annotations.coco.json", ROOT / "assets/aerial")
    assert dataset[0][1]["boxes"].shape == (15, 4)
    assert dataset[1][1]["boxes"].shape == (1, 4)


def test_reference_renderer_does_not_change_input_image() -> None:
    original = Image.new("RGB", (1600, 1200), (80, 90, 100))
    before = original.tobytes()
    preview = render_preview(original, [[10, 10, 60, 70], [1570, 1180, 1599, 1199]], scene="test")
    assert original.tobytes() == before
    assert preview.size == (1280, 1152)
    assert preview.getpixel((1, 1)) == (17, 27, 31)
    assert preview.getpixel((500, 500)) == (80, 90, 100)


def test_reference_previews_rebuild_without_full_dataset_or_model(tmp_path: Path) -> None:
    result = render_previews(ROOT / "assets/aerial/reference_annotations.coco.json", ROOT / "assets/aerial", tmp_path)
    assert result["kind"] == "reference_annotations_not_model_output"
    assert [entry["boxes"] for entry in result["previews"]] == [15, 1]
    assert all((tmp_path / entry["file"]).stat().st_size > 0 for entry in result["previews"])
    assert json.loads((tmp_path / "REFERENCE_PREVIEWS.json").read_text()) == result
