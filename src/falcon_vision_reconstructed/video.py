"""Video writer and synthetic demo helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np

from .models import Frame
from .sources import SyntheticFrameSource


class OutputVideoWriter:
    """Small video writer for reproducible demos.

    Frames are buffered in memory, which is appropriate for short portfolio and
    test videos. Large production video handling is outside Level B scope.
    """

    def __init__(self, path: str | Path, *, fps: float = 12.0) -> None:
        self.path = Path(path)
        self.name = f"video-writer:{self.path}"
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def write(self, frame: Frame, image: Image.Image) -> None:
        del frame
        self._frames.append(np.asarray(image.convert("RGB")))

    def close(self) -> None:
        if not self._frames:
            raise RuntimeError("No frames were written; refusing to create an empty video.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("OutputVideoWriter requires imageio and imageio-ffmpeg.") from exc
        iio.imwrite(self.path, self._frames, fps=self.fps)


def create_synthetic_demo_video(path: str | Path, *, frame_count: int = 24, fps: float = 12.0) -> Path:
    """Create a deterministic synthetic civilian demo video."""

    output_path = Path(path)
    writer = OutputVideoWriter(output_path, fps=fps)
    for frame in SyntheticFrameSource(frame_count=frame_count, fps=fps):
        writer.write(frame, frame.image)
    writer.close()
    return output_path

