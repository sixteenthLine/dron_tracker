"""Typed TOML configuration for the Level B pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    source_type: str = "video"
    detector_backend: str = "torchvision"
    tracker_backend: str = "iou"
    threshold: float = 0.5
    frame_skip: int = 0
    max_frames: int | None = None
    fps: float = 12.0
    device: str = "cpu"
    deterministic_demo: bool = False
    iou_threshold: float = 0.3
    max_missed: int = 5
    min_box_area: float = 25.0
    log_level: str = "INFO"

    @classmethod
    def from_toml(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise ValueError(f"Config file not found: {config_path}")
        with config_path.open("rb") as file:
            payload = tomllib.load(file)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PipelineConfig":
        pipeline = payload.get("pipeline", payload)
        if not isinstance(pipeline, dict):
            raise ValueError("Pipeline config must be a table.")

        input_path = pipeline.get("input_path")
        output_path = pipeline.get("output_path")
        if not isinstance(input_path, str) or not input_path:
            raise ValueError("Pipeline config requires string input_path.")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("Pipeline config requires string output_path.")

        config = cls(
            input_path=Path(input_path),
            output_path=Path(output_path),
            source_type=str(pipeline.get("source_type", "video")),
            detector_backend=str(pipeline.get("detector_backend", "torchvision")),
            tracker_backend=str(pipeline.get("tracker_backend", "iou")),
            threshold=float(pipeline.get("threshold", 0.5)),
            frame_skip=int(pipeline.get("frame_skip", 0)),
            max_frames=_optional_int(pipeline.get("max_frames")),
            fps=float(pipeline.get("fps", 12.0)),
            device=str(pipeline.get("device", "cpu")),
            deterministic_demo=bool(pipeline.get("deterministic_demo", False)),
            iou_threshold=float(pipeline.get("iou_threshold", 0.3)),
            max_missed=int(pipeline.get("max_missed", 5)),
            min_box_area=float(pipeline.get("min_box_area", 25.0)),
            log_level=str(pipeline.get("log_level", "INFO")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.source_type not in {"image", "video", "webcam", "synthetic"}:
            raise ValueError("source_type must be one of: image, video, webcam, synthetic")
        if self.detector_backend not in {"torchvision", "mock"}:
            raise ValueError("detector_backend must be one of: torchvision, mock")
        if self.tracker_backend not in {"iou", "legacy_goturn", "mock", "none"}:
            raise ValueError("tracker_backend must be one of: iou, legacy_goturn, mock, none")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if self.frame_skip < 0:
            raise ValueError("frame_skip must be zero or greater")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when set")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if self.max_missed < 0:
            raise ValueError("max_missed must be zero or greater")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
