"""Reusable frame processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

from .config import PipelineConfig
from .detector import create_detector_backend
from .interfaces import Detector, FrameSink, FrameSource, Tracker
from .logging_utils import configure_logging
from .models import FrameResult
from .sources import ImageFileSource, SyntheticFrameSource, VideoFileSource, WebcamSource
from .tracking import create_tracker_backend
from .video import OutputVideoWriter
from .visualization import draw_overlay


@dataclass(frozen=True)
class PipelineSummary:
    frames_seen: int
    frames_processed: int
    output_path: str | None
    average_latency_ms: float


class FrameProcessor:
    """Detector -> tracker -> visualization processor."""

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        *,
        sink: FrameSink | None = None,
        frame_skip: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        if frame_skip < 0:
            raise ValueError("frame_skip must be zero or greater")
        self.detector = detector
        self.tracker = tracker
        self.sink = sink
        self.frame_skip = frame_skip
        self.logger = logger or logging.getLogger(__name__)

    def process(self, source: FrameSource) -> PipelineSummary:
        frames_seen = 0
        frames_processed = 0
        latency_values: list[float] = []
        try:
            for frame in source:
                frames_seen += 1
                should_process = frame.index % (self.frame_skip + 1) == 0
                start = time.perf_counter()
                if should_process:
                    detections = self.detector.detect(frame.image, frame_index=frame.index)
                    tracks = self.tracker.update(detections, frame)
                    frames_processed += 1
                else:
                    detections = []
                    tracks = self.tracker.current_tracks()
                latency_ms = (time.perf_counter() - start) * 1000.0
                latency_values.append(latency_ms)
                result = FrameResult(
                    frame=frame,
                    detections=detections,
                    tracks=tracks,
                    processed=should_process,
                    latency_ms=latency_ms,
                )
                overlay = draw_overlay(frame.image, detections=result.detections, tracks=result.tracks)
                if self.sink is not None:
                    self.sink.write(frame, overlay)
                self._log_frame(result)
        finally:
            source.close()
            if self.sink is not None:
                self.sink.close()

        average_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
        output_path = getattr(self.sink, "path", None)
        return PipelineSummary(
            frames_seen=frames_seen,
            frames_processed=frames_processed,
            output_path=str(output_path) if output_path is not None else None,
            average_latency_ms=average_latency,
        )

    def _log_frame(self, result: FrameResult) -> None:
        self.logger.info(
            "frame_processed",
            extra={
                "event": {
                    "frame_index": result.frame.index,
                    "processed": result.processed,
                    "detections": len(result.detections),
                    "tracks": len(result.tracks),
                    "latency_ms": round(result.latency_ms, 3),
                }
            },
        )


def build_source(config: PipelineConfig) -> FrameSource:
    if config.source_type == "image":
        return ImageFileSource(config.input_path)
    if config.source_type == "video":
        return VideoFileSource(config.input_path, fps=config.fps, max_frames=config.max_frames)
    if config.source_type == "webcam":
        return WebcamSource(int(str(config.input_path)), fps=config.fps, max_frames=config.max_frames)
    if config.source_type == "synthetic":
        return SyntheticFrameSource(frame_count=config.max_frames or 24, fps=config.fps)
    raise ValueError(f"Unsupported source_type: {config.source_type}")


def build_processor(config: PipelineConfig) -> FrameProcessor:
    detector = create_detector_backend(
        config.detector_backend,
        threshold=config.threshold,
        max_detections=20,
        device=config.device,
    )
    tracker = create_tracker_backend(
        config.tracker_backend,
        iou_threshold=config.iou_threshold,
        max_missed=config.max_missed,
        min_box_area=config.min_box_area,
    )
    sink = OutputVideoWriter(config.output_path, fps=config.fps)
    return FrameProcessor(detector, tracker, sink=sink, frame_skip=config.frame_skip)


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    configure_logging(config.log_level)
    source = build_source(config)
    processor = build_processor(config)
    return processor.process(source)
