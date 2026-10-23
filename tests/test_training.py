import ast
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from falcon_vision_reconstructed import training
from falcon_vision_reconstructed.training import TrainingConfig, select_records, tiled_predict
from falcon_vision_reconstructed.training_data import TrainingTiles, clipped_boxes, grid_starts, prepare_data


def write_export(root: Path, *, empty: bool = True) -> Path:
    images = root / "train"
    images.mkdir(parents=True)
    records = []
    annotations = []
    groups = ["ONE", "ONE", "VAL", "TEST"] + (["EMPTY"] if empty else [])
    for index, group in enumerate(groups):
        name = f"train_{group}_{index:04d}.jpg"
        Image.new("RGB", (32, 32), (index * 31, 40, 60)).save(images / name)
        records.append({"id": index, "file_name": name, "width": 32, "height": 32})
        if group != "EMPTY":
            annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [4, 4, 12, 12]})
    (root / "_annotations.coco.json").write_text(json.dumps({
        "images": records, "annotations": annotations, "categories": [{"id": 1, "name": "human"}],
    }))
    return root


def test_prepare_separates_groups_and_excludes_unverified_empty_images(tmp_path: Path) -> None:
    root = write_export(tmp_path / "data")
    result = prepare_data(root, tmp_path / "prepared", val_groups=("VAL",), test_groups=("TEST",))
    manifest = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert result["unannotated_excluded"] == 1
    assert [len(manifest["splits"][key]) for key in ("train", "val", "test")] == [2, 1, 1]
    groups = [{r["group"] for r in manifest["splits"][key]} for key in ("train", "val", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_prepare_rejects_exact_duplicate_crossing_splits(tmp_path: Path) -> None:
    root = write_export(tmp_path / "data")
    (root / "train/train_VAL_0002.jpg").write_bytes((root / "train/train_ONE_0000.jpg").read_bytes())
    with pytest.raises(ValueError, match="duplicate image crosses"):
        prepare_data(root, tmp_path / "prepared", val_groups=("VAL",), test_groups=("TEST",))


def test_grid_covers_right_edge_and_rejects_invalid_overlap() -> None:
    assert grid_starts(100, 64, 16) == [0, 36]
    assert grid_starts(32, 64, 16) == [0]
    with pytest.raises(ValueError):
        grid_starts(100, 64, 64)


def test_box_clipping_translates_and_filters_mostly_missing_objects() -> None:
    assert clipped_boxes([[10, 10, 30, 30], [0, 0, 12, 12]], (15, 10, 40, 40)) == [[0, 0, 15, 20]]


def test_tiles_preserve_native_pixel_scale(tmp_path: Path) -> None:
    Image.new("RGB", (100, 80)).save(tmp_path / "one.png")
    records = [{"id": 4, "file_name": "one.png", "width": 100, "height": 80, "boxes": [[10, 10, 30, 30]]}]
    data = TrainingTiles(records, tmp_path, size=64, overlap=16)
    image, target = data[0]
    assert image.shape == (3, 64, 64)
    assert target["boxes"].tolist() == [[10, 10, 30, 30]]


def test_pilot_selection_covers_groups_deterministically() -> None:
    records = [{"group": group, "id": index} for index, group in enumerate(["a", "a", "b", "b"])]
    selected = select_records(records, 2, 42)
    assert {r["group"] for r in selected} == {"a", "b"}
    assert selected == select_records(records, 2, 42)


class FakeDetector(torch.nn.Module):
    def __init__(self, *, empty: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.backbone = torch.nn.Identity()
        self.empty = empty

    def forward(self, images, targets=None):
        if targets is not None:
            return {"loss_classifier": (self.weight - images[0].mean()).square()}
        boxes = torch.empty((0, 4)) if self.empty else torch.tensor([[4., 4., 16., 16.]])
        return [{"boxes": boxes, "scores": torch.full((len(boxes),), 0.9),
                 "labels": torch.ones(len(boxes), dtype=torch.int64)} for _ in images]


def config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(manifest=str(tmp_path / "prepared/manifest.json"), image_root=str(tmp_path / "data/train"),
                          output_dir=str(tmp_path / "run"), image_size=32, tile_size=32, overlap=8,
                          epochs=3, max_steps=1, checkpoint_every=1, device="cpu")


def test_tiled_predictions_restore_full_image_coordinates(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    output = tiled_predict(FakeDetector(), Image.new("RGB", (56, 32)), cfg, torch.device("cpu"))
    assert output["boxes"].tolist() == [[4., 4., 16., 16.], [28., 4., 40., 16.]]


@pytest.mark.parametrize("empty, expected", [(False, 1.0), (True, 0.0)])
def test_coco_metrics_handle_perfect_and_empty_predictions(tmp_path: Path, empty: bool, expected: float) -> None:
    pytest.importorskip("pycocotools")
    Image.new("RGB", (32, 32)).save(tmp_path / "one.png")
    records = [{"id": 4, "file_name": "one.png", "width": 32, "height": 32, "boxes": [[4, 4, 16, 16]]}]
    result = training.evaluate(FakeDetector(empty=empty), records, tmp_path, config(tmp_path), torch.device("cpu"))
    assert result["map"] == pytest.approx(expected)


def test_real_optimizer_checkpoint_and_resume_without_network(tmp_path: Path, monkeypatch) -> None:
    root = write_export(tmp_path / "data")
    prepare_data(root, tmp_path / "prepared", val_groups=("VAL",), test_groups=("TEST",))
    monkeypatch.setattr(training, "build_training_model", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: {"map": 0.0, "images": 1})
    cfg = config(tmp_path)
    first = training.train(cfg)
    before = training.load_checkpoint(Path(cfg.output_dir) / "last.pt")
    cfg.max_steps = 2
    second = training.train(cfg, Path(cfg.output_dir) / "last.pt")
    after = training.load_checkpoint(Path(cfg.output_dir) / "last.pt")
    assert first["global_steps"] == 1
    assert second["global_steps"] == 2 and second["new_steps"] == 1
    assert not torch.equal(before["model"]["weight"], after["model"]["weight"])
    cfg.learning_rate *= 2
    with pytest.raises(ValueError, match="learning_rate"):
        training.train(cfg, Path(cfg.output_dir) / "last.pt")


def test_training_config_rejects_unbounded_invalid_values(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.learning_rate = float("nan")
    with pytest.raises(ValueError, match="learning_rate"):
        cfg.validate()


def test_checkpoint_model_uses_binary_head_and_frozen_backbone_norms(tmp_path: Path) -> None:
    from torchvision.ops.misc import FrozenBatchNorm2d

    model = training.build_training_model(config(tmp_path), pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert any(isinstance(layer, FrozenBatchNorm2d) for layer in model.backbone.modules())
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.backbone.modules())
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.roi_heads.parameters())


def test_colab_notebook_source_parses_without_claiming_cloud_execution() -> None:
    notebook = Path(__file__).resolve().parents[1] / "notebooks/Aerial_Detection_Training.ipynb"
    data = json.loads(notebook.read_text())
    cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert data["nbformat"] == 4 and cells
    for cell in cells:
        ast.parse("".join(cell["source"]))
        assert cell["execution_count"] is None and not cell["outputs"]
