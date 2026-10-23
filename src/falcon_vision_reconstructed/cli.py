"""Command-line interface for the safe detector and Level B pipeline demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

from .config import PipelineConfig
from .detector import MODEL_NAME, predict_image
from .pipeline import run_pipeline
from .ros_bridge import is_ros2_available, run_ros_node, topic_contract
from .video import create_synthetic_demo_video
from .visualization import draw_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aerial Object Detection: offline civilian CV demos.")
    subparsers = parser.add_subparsers(dest="command")

    _add_infer_image_parser(subparsers)
    _add_infer_video_parser(subparsers)
    _add_run_pipeline_parser(subparsers)
    _add_create_demo_video_parser(subparsers)
    _add_ros_node_parser(subparsers)
    _add_validate_parser(subparsers)

    return parser


def _add_infer_image_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("infer-image", help="Run detector inference on one image.")
    parser.add_argument("--input", required=True, help="Path to an input image.")
    parser.add_argument("--output", required=True, help="Path for the output image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold. Default: 0.5.")
    parser.add_argument("--max-detections", type=int, default=20, help="Maximum detections to draw.")
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu.")


def _add_infer_video_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("infer-video", help="Run detector/tracker pipeline on a video file.")
    parser.add_argument("--input", required=True, help="Path to an input video.")
    parser.add_argument("--output", required=True, help="Path for the output video.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--frame-skip", type=int, default=0, help="Number of frames to skip between detections.")
    parser.add_argument("--detector", choices=["torchvision", "mock"], default="mock")
    parser.add_argument("--tracker", choices=["iou", "none", "mock", "legacy_goturn"], default="iou")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")


def _add_run_pipeline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run-pipeline", help="Run the pipeline from a TOML config.")
    parser.add_argument("--config", required=True, help="Path to a TOML pipeline config.")


def _add_create_demo_video_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("create-demo-video", help="Create a deterministic synthetic civilian demo video.")
    parser.add_argument("--output", default="outputs/synthetic_civilian_input.mp4")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=12.0)


def _add_ros_node_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("ros-node", help="Start the optional ROS2 camera CV bridge.")
    parser.add_argument("--check-only", action="store_true", help="Only report whether ROS2 is available.")


def _add_validate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("validate", help="Print optional component availability and safe topic contract.")


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0].startswith("--"):
        return _run_infer_image(args_list)

    parser = build_parser()
    args = parser.parse_args(args_list)
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "infer-image":
        return _run_infer_image_from_args(args)
    if args.command == "infer-video":
        config = PipelineConfig(
            input_path=Path(args.input),
            output_path=Path(args.output),
            source_type="video",
            detector_backend=args.detector,
            tracker_backend=args.tracker,
            threshold=args.threshold,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            fps=args.fps,
            device=args.device,
            log_level=args.log_level,
        )
        return _run_pipeline_config(config)
    if args.command == "run-pipeline":
        return _run_pipeline_config(PipelineConfig.from_toml(args.config))
    if args.command == "create-demo-video":
        output = create_synthetic_demo_video(args.output, frame_count=args.frames, fps=args.fps)
        print(f"Created synthetic demo video: {output}")
        return 0
    if args.command == "ros-node":
        if args.check_only:
            print(json.dumps({"ros2_available": is_ros2_available(), "topics": topic_contract()}, sort_keys=True))
            return 0
        return run_ros_node()
    if args.command == "validate":
        print(json.dumps({"ros2_available": is_ros2_available(), "topics": topic_contract()}, sort_keys=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def _run_infer_image(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run detector inference on one image.")
    _add_image_args(parser)
    args = parser.parse_args(argv)
    return _run_infer_image_from_args(args)


def _add_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Path to an input image.")
    parser.add_argument("--output", required=True, help="Path for the output image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold. Default: 0.5.")
    parser.add_argument("--max-detections", type=int, default=20, help="Maximum detections to draw.")
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu.")


def _run_infer_image_from_args(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        print(f"Input image not found: {input_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(input_path) as image_file:
            image = image_file.convert("RGB")
        detections = predict_image(
            image,
            threshold=args.threshold,
            max_detections=args.max_detections,
            device=args.device,
        )
        output = draw_detections(image, detections)
        output.save(output_path)
    except Exception as exc:
        print(f"Image inference failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Saved {output_path} with {len(detections.boxes)} detections "
        f"at threshold {args.threshold:.2f} using {MODEL_NAME}."
    )
    return 0


def _run_pipeline_config(config: PipelineConfig) -> int:
    try:
        summary = run_pipeline(config)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "frames_seen": summary.frames_seen,
                "frames_processed": summary.frames_processed,
                "average_latency_ms": round(summary.average_latency_ms, 3),
                "output_path": summary.output_path,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
