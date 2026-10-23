"""Frame sources for images, videos, and optional webcams."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image
import numpy as np

from .models import Frame


class ImageFileSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = f"image:{self.path}"
        self._emitted = False
        if not self.path.is_file():
            raise ValueError(f"Input image not found: {self.path}")

    def __iter__(self) -> "ImageFileSource":
        return self

    def __next__(self) -> Frame:
        if self._emitted:
            raise StopIteration
        self._emitted = True
        with Image.open(self.path) as image_file:
            image = image_file.convert("RGB")
        return Frame(index=0, timestamp_sec=0.0, image=image)

    def close(self) -> None:
        return None


class VideoFileSource:
    def __init__(self, path: str | Path, *, fps: float | None = None, max_frames: int | None = None) -> None:
        self.path = Path(path)
        self.name = f"video:{self.path}"
        self.fps = fps or 30.0
        self.max_frames = max_frames
        if not self.path.is_file():
            raise ValueError(f"Input video not found: {self.path}")
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("VideoFileSource requires imageio and imageio-ffmpeg.") from exc
        self._iio = iio
        self._iterator: Iterator[np.ndarray] | None = None
        self._index = 0

    def __iter__(self) -> "VideoFileSource":
        self._iterator = iter(self._iio.imiter(self.path))
        return self

    def __next__(self) -> Frame:
        if self.max_frames is not None and self._index >= self.max_frames:
            raise StopIteration
        if self._iterator is None:
            self.__iter__()
        assert self._iterator is not None
        array = next(self._iterator)
        if array.ndim == 2:
            image = Image.fromarray(array).convert("RGB")
        else:
            image = Image.fromarray(array[:, :, :3]).convert("RGB")
        frame = Frame(index=self._index, timestamp_sec=self._index / self.fps, image=image)
        self._index += 1
        return frame

    def close(self) -> None:
        self._iterator = None


class WebcamSource:
    """Optional webcam source for local demos only.

    This source does not control any device beyond reading frames from a camera.
    """

    def __init__(self, camera_index: int = 0, *, fps: float = 30.0, max_frames: int | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("WebcamSource requires the optional opencv dependency.") from exc
        self._cv2 = cv2
        self.name = f"webcam:{camera_index}"
        self.fps = fps
        self.max_frames = max_frames
        self._index = 0
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise ValueError(f"Could not open webcam index {camera_index}.")

    def __iter__(self) -> "WebcamSource":
        return self

    def __next__(self) -> Frame:
        if self.max_frames is not None and self._index >= self.max_frames:
            raise StopIteration
        ok, frame_bgr = self._capture.read()
        if not ok:
            raise StopIteration
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).convert("RGB")
        frame = Frame(index=self._index, timestamp_sec=self._index / self.fps, image=image)
        self._index += 1
        return frame

    def close(self) -> None:
        self._capture.release()


class SyntheticFrameSource:
    """Deterministic in-memory synthetic source for tests and safe demos."""

    def __init__(self, *, frame_count: int = 24, width: int = 320, height: int = 240, fps: float = 12.0) -> None:
        self.name = "synthetic:civilian-demo"
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.fps = fps
        self._index = 0

    def __iter__(self) -> "SyntheticFrameSource":
        self._index = 0
        return self

    def __next__(self) -> Frame:
        if self._index >= self.frame_count:
            raise StopIteration
        from PIL import ImageDraw

        image = Image.new("RGB", (self.width, self.height), color=(228, 236, 238))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, self.height - 48, self.width, self.height), fill=(70, 112, 86))
        draw.rectangle((24, 32, 92, 84), fill=(180, 190, 195))
        x1 = 40 + self._index * 5
        y1 = 132
        draw.rectangle((x1, y1, x1 + 50, y1 + 28), fill=(44, 94, 168))
        draw.ellipse((x1 + 6, y1 + 24, x1 + 18, y1 + 36), fill=(28, 28, 28))
        draw.ellipse((x1 + 34, y1 + 24, x1 + 46, y1 + 36), fill=(28, 28, 28))
        frame = Frame(index=self._index, timestamp_sec=self._index / self.fps, image=image)
        self._index += 1
        return frame

    def close(self) -> None:
        return None

