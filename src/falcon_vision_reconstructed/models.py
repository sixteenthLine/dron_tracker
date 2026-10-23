"""Shared data models for the safe Level B pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: BBox
    label: str
    label_id: int
    score: float


@dataclass(frozen=True)
class Track:
    track_id: int
    box: BBox
    label: str
    label_id: int
    score: float
    age: int
    missed: int


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_sec: float
    image: Image.Image


@dataclass(frozen=True)
class FrameResult:
    frame: Frame
    detections: list[Detection]
    tracks: list[Track]
    processed: bool
    latency_ms: float

