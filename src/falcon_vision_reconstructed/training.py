"""New offline fine-tuning experiments, separate from the Level A/B demos."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
import random
import time
import tomllib
from typing import Any

from PIL import Image
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import to_tensor

from .detector import DetectionResult
from .training_data import TrainingTiles, file_sha256, grid_starts, prepare_data
from .visualization import draw_detections


@dataclass
class TrainingConfig:
    manifest: str
    image_root: str
    output_dir: str
    scope: str = "pilot"
    device: str = "auto"
    image_size: int = 512
    tile_size: int = 768
    overlap: int = 128
    batch_size: int = 1
    inference_batch_size: int = 1
    epochs: int = 2
    max_steps: int | None = None
    max_minutes: float | None = None
    train_limit: int | None = None
    val_limit: int | None = None
    train_mode: str = "heads"
    learning_rate: float = 0.001
    seed: int = 42
    checkpoint_every: int = 20
    patience: int = 5

    def validate(self) -> None:
        if self.scope not in {"pilot", "full"} or self.train_mode not in {"heads", "full"}:
            raise ValueError("scope must be pilot/full and train_mode must be heads/full")
        for key in ("image_size", "tile_size", "batch_size", "inference_batch_size", "epochs", "checkpoint_every", "patience"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("max_steps", "train_limit", "val_limit"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{key} must be positive when set")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.max_minutes is not None and (not math.isfinite(self.max_minutes) or self.max_minutes <= 0):
            raise ValueError("max_minutes must be finite and positive")


def select_records(records: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Spread a bounded pilot across filename groups instead of taking adjacent frames."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        raise ValueError("Image limit must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        for group in sorted(groups):
            if groups[group]:
                selected.append(groups[group].pop())
                if len(selected) == limit:
                    return selected
    return selected


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        # MPS remains opt-in: full-scene inference stalled in the local backend.
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select a GPU runtime or use cpu/mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def build_training_model(config: TrainingConfig, *, pretrained: bool) -> torch.nn.Module:
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=None, min_size=config.image_size, max_size=config.image_size,
    )
    # Loading without upstream weights must retain the pretrained backbone's
    # FrozenBatchNorm layout and epsilon, not create trainable BatchNorm layers.
    if not pretrained:
        def freeze_norms(module: torch.nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, torch.nn.BatchNorm2d):
                    module.add_module(name, FrozenBatchNorm2d(child.num_features))
                else:
                    freeze_norms(child)
        freeze_norms(model.backbone)
    old = model.roi_heads.box_predictor
    predictor = FastRCNNPredictor(old.cls_score.in_features, 2)
    if pretrained:
        # COCO's background and person rows initialize the binary experiment.
        with torch.no_grad():
            predictor.cls_score.weight.copy_(old.cls_score.weight[:2])
            predictor.cls_score.bias.copy_(old.cls_score.bias[:2])
            predictor.bbox_pred.weight.copy_(old.bbox_pred.weight[:8])
            predictor.bbox_pred.bias.copy_(old.bbox_pred.bias[:8])
    model.roi_heads.box_predictor = predictor
    if config.train_mode == "heads":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.roi_heads.parameters():
            parameter.requires_grad_(True)
    return model


@torch.inference_mode()
def tiled_predict(model: torch.nn.Module, image: Image.Image, config: TrainingConfig, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    windows = [(left, top, min(left + config.tile_size, image.width), min(top + config.tile_size, image.height))
               for top in grid_starts(image.height, config.tile_size, config.overlap)
               for left in grid_starts(image.width, config.tile_size, config.overlap)]
    boxes, scores = [], []
    for offset in range(0, len(windows), config.inference_batch_size):
        batch = windows[offset:offset + config.inference_batch_size]
        inputs = [to_tensor(image.crop(window).convert("RGB")).to(device) for window in batch]
        predictions = model(inputs)
        for window, prediction in zip(batch, predictions):
            keep = (prediction["labels"] == 1) & (prediction["scores"] >= 0.05)
            tile_boxes = prediction["boxes"][keep].detach().cpu()
            tile_boxes += torch.tensor([window[0], window[1], window[0], window[1]])
            boxes.append(tile_boxes)
            scores.append(prediction["scores"][keep].detach().cpu())
    all_boxes = torch.cat(boxes) if boxes else torch.empty((0, 4))
    all_scores = torch.cat(scores) if scores else torch.empty(0)
    keep = nms(all_boxes, all_scores, 0.5)[:100]
    return {"boxes": all_boxes[keep], "scores": all_scores[keep], "labels": torch.ones(len(keep), dtype=torch.int64)}


def evaluate(model: torch.nn.Module, records: list[dict[str, Any]], image_root: Path,
             config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Evaluation needs the training extra: pip install -r requirements-training.txt") from exc
    annotations, predictions = [], []
    for index, record in enumerate(records):
        for box in record["boxes"]:
            x, y, right, bottom = box
            annotations.append({"id": len(annotations) + 1, "image_id": record["id"], "category_id": 1,
                                "bbox": [x, y, right - x, bottom - y], "area": (right - x) * (bottom - y), "iscrowd": 0})
        with Image.open(image_root / record["file_name"]) as image:
            output = tiled_predict(model, image.convert("RGB"), config, device)
        for box, score in zip(output["boxes"].tolist(), output["scores"].tolist()):
            predictions.append({"image_id": record["id"], "category_id": 1,
                                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "score": score})
        print(json.dumps({"event": "evaluation_image", "completed": index + 1, "total": len(records)}), flush=True)
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {"info": {}, "images": [{k: r[k] for k in ("id", "width", "height")} for r in records],
                                "categories": [{"id": 1, "name": "human"}], "annotations": annotations}
        ground_truth.createIndex()
        detections = COCO()
        detections.dataset = {**ground_truth.dataset, "annotations": []}
        detections.createIndex()
        if predictions:
            detections = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = [record["id"] for record in records]
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"map": float(evaluator.stats[0]), "map_50": float(evaluator.stats[1]),
            "map_75": float(evaluator.stats[2]), "mar_100": float(evaluator.stats[8]),
            "images": len(records), "ground_truth_boxes": len(annotations),
            "predictions_at_0_5": sum(p["score"] >= 0.5 for p in predictions),
            "image_ids": [r["id"] for r in records],
            "protocol": "fixed full-frame grid, global NMS, COCO bbox evaluation; positive-only subset"}


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    config: TrainingConfig, manifest_hash: str, progress: dict[str, Any]) -> None:
    checkpoint = {"format_version": 1, "architecture": "fasterrcnn_mobilenet_v3_large_fpn_binary",
                  "config": asdict(config), "manifest_sha256": manifest_hash,
                  "trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                  "model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
                  "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), **progress}
    device_type = next(model.parameters()).device.type
    if device_type == "cuda":
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    if device_type == "mps":
        checkpoint["mps_rng"] = torch.mps.get_rng_state()
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "fasterrcnn_mobilenet_v3_large_fpn_binary":
        raise ValueError("Unsupported reconstruction checkpoint format")
    return checkpoint


def train(config: TrainingConfig, resume: Path | None = None) -> dict[str, Any]:
    config.validate()
    started = time.monotonic()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(4)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    if (output / "last.pt").exists() and resume is None:
        raise ValueError("Run already has last.pt; use --resume or select a new output_dir")
    manifest_path = Path(config.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = file_sha256(manifest_path)
    train_records = select_records(manifest["splits"]["train"], config.train_limit, config.seed)
    val_records = select_records(manifest["splits"]["val"], config.val_limit, config.seed)
    tiles = TrainingTiles(train_records, Path(config.image_root), size=config.tile_size, overlap=config.overlap)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume manifest differs from checkpoint")
        for key in ("image_size", "tile_size", "overlap", "seed", "train_mode", "train_limit", "batch_size", "val_limit", "learning_rate"):
            if checkpoint["config"][key] != getattr(config, key):
                raise ValueError(f"Resume config differs in {key}")
    model = build_training_model(config, pretrained=checkpoint is None).to(device)
    if checkpoint:
        trainable_names = set(checkpoint["trainable_names"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable_names)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    progress: dict[str, Any] = {"epoch": 0, "next_position": 0, "global_step": 0, "best_map": -1.0, "bad_epochs": 0}
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    def log(event: dict[str, Any]) -> None:
        line = json.dumps(event, allow_nan=False)
        print(line, flush=True)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(line + "\n")

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for key in progress:
            progress[key] = checkpoint[key]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "mps" and "mps_rng" in checkpoint:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        if device.type == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        baseline = checkpoint.get("baseline_metrics")
        if progress["epoch"] >= config.epochs or config.max_steps is not None and progress["global_step"] >= config.max_steps:
            raise ValueError("Checkpoint already reached this run's limit; increase epochs/max_steps to continue")
    else:
        baseline = evaluate(model, val_records, Path(config.image_root), config, device)
        (output / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2) + "\n")
        progress["best_map"] = baseline["map"]
        progress["baseline_metrics"] = baseline
        save_checkpoint(output / "baseline.pt", model, optimizer, config, manifest_hash, progress)
        save_checkpoint(output / "best.pt", model, optimizer, config, manifest_hash, progress)
    progress["baseline_metrics"] = baseline
    initial_step = progress["global_step"]
    log({"event": "training_start", "device": str(device), "scope": config.scope,
         "training_images": len(train_records), "positive_tiles": len(tiles), "validation_images": len(val_records),
         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "resume_step": initial_step})
    stop_reason = "epochs_complete"
    losses: list[float] = []
    final_metrics = baseline
    for epoch in range(progress["epoch"], config.epochs):
        order = torch.randperm(len(tiles), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
        position = progress["next_position"] if epoch == progress["epoch"] else 0
        model.train()
        if config.train_mode == "heads":
            model.backbone.eval()
        for start in range(position, len(order), config.batch_size):
            if config.max_steps is not None and progress["global_step"] >= config.max_steps:
                stop_reason = "step_limit"
                break
            if config.max_minutes is not None and time.monotonic() - started >= config.max_minutes * 60:
                stop_reason = "time_limit"
                break
            batch = [tiles[index] for index in order[start:start + config.batch_size]]
            images = [image.to(device) for image, _ in batch]
            targets = [{key: value.to(device) for key, value in target.items()} for _, target in batch]
            optimizer.zero_grad(set_to_none=True)
            components = model(images, targets)
            loss = sum(components.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stopping without claiming a successful training run")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            progress.update(epoch=epoch, next_position=min(start + config.batch_size, len(order)), global_step=progress["global_step"] + 1)
            log({"event": "training_step", "epoch": epoch + 1, "step": progress["global_step"],
                 "loss": value, "components": {key: float(item.detach().cpu()) for key, item in components.items()},
                 "elapsed_seconds": round(time.monotonic() - started, 2)})
            if progress["global_step"] % config.checkpoint_every == 0:
                save_checkpoint(output / "last.pt", model, optimizer, config, manifest_hash, progress)
            if config.max_steps is not None and progress["global_step"] >= config.max_steps:
                stop_reason = "step_limit"
                break
        final_metrics = evaluate(model, val_records, Path(config.image_root), config, device)
        log({"event": "validation", "step": progress["global_step"], **final_metrics})
        if final_metrics["map"] > progress["best_map"]:
            progress.update(best_map=final_metrics["map"], bad_epochs=0)
            save_checkpoint(output / "best.pt", model, optimizer, config, manifest_hash, progress)
        else:
            progress["bad_epochs"] += 1
        if progress["next_position"] >= len(order):
            progress.update(epoch=epoch + 1, next_position=0)
        save_checkpoint(output / "last.pt", model, optimizer, config, manifest_hash, progress)
        if stop_reason != "epochs_complete":
            break
        if progress["bad_epochs"] >= config.patience:
            stop_reason = "early_stopping"
            break
    summary = {"scope": config.scope, "status": stop_reason, "device": str(device),
               "global_steps": progress["global_step"], "new_steps": progress["global_step"] - initial_step,
               "elapsed_seconds": round(time.monotonic() - started, 2), "mean_training_loss": sum(losses) / len(losses) if losses else None,
               "baseline": baseline, "final": final_metrics, "best_map": progress["best_map"],
               "training_images": len(train_records), "training_tiles": len(tiles), "validation_images": len(val_records),
               "test_evaluated": False, "historical_results": False,
               "caveat": "New experiment; positive-only validation subset, not a historical or full-dataset benchmark"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    log({"event": "training_finished", **summary})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--image-root")
    fit.add_argument("--manifest")
    fit.add_argument("--output-dir")
    fit.add_argument("--device")
    fit.add_argument("--max-steps", type=int)
    for name in ("evaluate", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--image-root", type=Path, required=True)
            command.add_argument("--split", choices=["val", "test"], default="val")
            command.add_argument("--limit", type=int)
        else:
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare_data(args.dataset, args.output), indent=2))
        elif args.command == "train":
            values = tomllib.loads(args.config.read_text())["training"]
            for key in ("image_root", "manifest", "output_dir", "device", "max_steps"):
                if getattr(args, key) is not None:
                    values[key] = getattr(args, key)
            train(TrainingConfig(**values), args.resume)
        else:
            checkpoint = load_checkpoint(args.checkpoint)
            config = TrainingConfig(**checkpoint["config"])
            device = resolve_device(args.device)
            model = build_training_model(config, pretrained=False)
            model.load_state_dict(checkpoint["model"])
            model.to(device).eval()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.command == "evaluate":
                if file_sha256(args.manifest) != checkpoint["manifest_sha256"]:
                    raise ValueError("Evaluation manifest differs from training checkpoint")
                manifest = json.loads(args.manifest.read_text())
                records = select_records(manifest["splits"][args.split], args.limit, config.seed)
                metrics = evaluate(model, records, args.image_root, config, device)
                args.output.write_text(json.dumps(metrics, indent=2) + "\n")
                print(json.dumps(metrics))
            else:
                if not 0 <= args.threshold <= 1:
                    raise ValueError("threshold must be between zero and one")
                with Image.open(args.input) as source:
                    image = source.convert("RGB")
                predictions = tiled_predict(model, image, config, device)
                keep = predictions["scores"] >= args.threshold
                boxes = predictions["boxes"][keep].tolist()
                result = DetectionResult(boxes=[tuple(box) for box in boxes], labels=["human"] * len(boxes),
                                         label_ids=[1] * len(boxes), scores=predictions["scores"][keep].tolist())
                draw_detections(image, result).save(args.output)
                print(json.dumps({"output": str(args.output), "detections": len(boxes), "checkpoint_step": checkpoint["global_step"]}))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Training command failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
