from pathlib import Path

from PIL import Image

from falcon_vision_reconstructed.config import PipelineConfig
from falcon_vision_reconstructed.detector import MockDetector
from falcon_vision_reconstructed.models import Frame
from falcon_vision_reconstructed.pipeline import FrameProcessor
from falcon_vision_reconstructed.sources import SyntheticFrameSource, VideoFileSource
from falcon_vision_reconstructed.tracking import IoUTracker
from falcon_vision_reconstructed.video import create_synthetic_demo_video


class MemorySink:
    name = "memory-sink"

    def __init__(self) -> None:
        self.frames: list[tuple[int, Image.Image]] = []
        self.closed = False

    def write(self, frame: Frame, image: Image.Image) -> None:
        self.frames.append((frame.index, image.copy()))

    def close(self) -> None:
        self.closed = True


def test_pipeline_processes_synthetic_frames_with_mock_detector_and_iou_tracker() -> None:
    source = SyntheticFrameSource(frame_count=4, width=160, height=120, fps=8)
    sink = MemorySink()
    processor = FrameProcessor(MockDetector(), IoUTracker(iou_threshold=0.2), sink=sink)

    summary = processor.process(source)

    assert summary.frames_seen == 4
    assert summary.frames_processed == 4
    assert len(sink.frames) == 4
    assert sink.closed is True
    assert summary.average_latency_ms >= 0.0


def test_pipeline_frame_skip_reuses_current_tracks_without_detection() -> None:
    source = SyntheticFrameSource(frame_count=5, width=160, height=120, fps=8)
    sink = MemorySink()
    processor = FrameProcessor(MockDetector(), IoUTracker(iou_threshold=0.2), sink=sink, frame_skip=1)

    summary = processor.process(source)

    assert summary.frames_seen == 5
    assert summary.frames_processed == 3
    assert [index for index, _ in sink.frames] == [0, 1, 2, 3, 4]


def test_pipeline_config_from_mapping_validates_options() -> None:
    config = PipelineConfig.from_mapping(
        {
            "pipeline": {
                "input_path": "input.mp4",
                "output_path": "output.mp4",
                "source_type": "video",
                "detector_backend": "mock",
                "tracker_backend": "iou",
                "threshold": 0.4,
                "frame_skip": 2,
                "max_frames": 5,
            }
        }
    )

    assert config.input_path == Path("input.mp4")
    assert config.detector_backend == "mock"
    assert config.frame_skip == 2
    assert config.max_frames == 5


def test_pipeline_config_accepts_synthetic_demo_source() -> None:
    config = PipelineConfig.from_mapping(
        {
            "input_path": "synthetic:civilian-demo",
            "output_path": "output.mp4",
            "source_type": "synthetic",
            "detector_backend": "mock",
            "tracker_backend": "iou",
            "deterministic_demo": True,
        }
    )

    assert config.source_type == "synthetic"
    assert config.deterministic_demo is True


def test_create_and_read_synthetic_demo_video(tmp_path: Path) -> None:
    output = create_synthetic_demo_video(tmp_path / "demo.mp4", frame_count=3, fps=6)

    source = VideoFileSource(output, fps=6)
    frames = list(source)
    source.close()

    assert output.is_file()
    assert len(frames) == 3
    assert frames[0].image.size == (320, 240)
