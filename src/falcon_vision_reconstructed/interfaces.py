"""Explicit interfaces for reusable pipeline components."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from .models import Detection, Frame, Track


class FrameSource(Protocol):
    name: str

    def __iter__(self) -> "FrameSource":
        ...

    def __next__(self) -> Frame:
        ...

    def close(self) -> None:
        ...


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image, *, frame_index: int = 0) -> list[Detection]:
        ...


class Tracker(Protocol):
    name: str

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        ...

    def current_tracks(self) -> list[Track]:
        ...


class FrameSink(Protocol):
    name: str

    def write(self, frame: Frame, image: Image.Image) -> None:
        ...

    def close(self) -> None:
        ...

