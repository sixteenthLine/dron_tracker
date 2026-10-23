"""Render documentation previews from real demo outputs, without running models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont, ImageOps


def comparison(
    left: Image.Image,
    right: Image.Image,
    *,
    labels: tuple[str, str],
    cell_size: tuple[int, int],
) -> Image.Image:
    width, height = cell_size
    canvas = Image.new("RGB", (width * 2 + 24, height + 44), (245, 247, 249))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for index, (image, label) in enumerate(zip((left, right), labels)):
        x = 8 + index * (width + 8)
        draw.text((x, 10), label, font=font, fill=(26, 36, 43))
        fitted = ImageOps.contain(image.convert("RGB"), cell_size)
        canvas.paste(fitted, (x + (width - fitted.width) // 2, 36 + (height - fitted.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, action="append")
    parser.add_argument("--detected-image", type=Path, action="append")
    parser.add_argument("--raw-video", type=Path, default=Path("outputs/synthetic_civilian_input.mp4"))
    parser.add_argument("--tracked-video", type=Path, default=Path("outputs/synthetic_civilian_tracked.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/media"))
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    inputs = args.input_image or [Path("assets/aerial/park.jpg"), Path("assets/aerial/terrain.jpg")]
    detected_images = args.detected_image or [Path("outputs/aerial_park_detected.jpg"), Path("outputs/aerial_terrain_detected.jpg")]
    if len(inputs) != len(detected_images):
        parser.error("Provide equal numbers of input and detected images")
    for path in (*inputs, *detected_images, args.raw_video, args.tracked_video):
        if not path.is_file():
            parser.error(f"Required demo output or input not found: {path}")
    if not 1 <= args.fps <= 100:
        parser.error("fps must be between 1 and 100")

    previews = []
    for source_path, detected_path in zip(inputs, detected_images):
        with Image.open(source_path) as original, Image.open(detected_path) as detected:
            previews.append(comparison(
                original, detected,
                labels=("HERIDAL / AERIAL INPUT", "PRETRAINED DETECTOR"),
                cell_size=(480, 360),
            ))

    raw_frames = list(iio.imiter(args.raw_video))
    tracked_frames = list(iio.imiter(args.tracked_video))
    if not raw_frames or len(raw_frames) != len(tracked_frames):
        parser.error("The raw and tracked demo videos must have equal, nonzero frame counts")
    frames = [
        comparison(
            Image.fromarray(raw), Image.fromarray(tracked),
            labels=("INPUT / SYNTHETIC", "MOCK DETECTION + IoU ID"),
            cell_size=(320, 240),
        ).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for raw, tracked in zip(raw_frames, tracked_frames)
    ]
    # GIF durations are stored in 10 ms ticks; distribute rounding across frames.
    durations = [
        (round((index + 1) * 100 / args.fps) - round(index * 100 / args.fps)) * 10
        for index in range(len(frames))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, preview in enumerate(previews):
        name = "detection.jpg" if index == 0 else f"detection_{index + 1}.jpg"
        preview.save(args.output_dir / name, quality=90, optimize=True)
    if len(inputs) >= 2:
        with Image.open(inputs[0]) as left, Image.open(inputs[1]) as right:
            gallery = comparison(
                left, right,
                labels=(f"HERIDAL / {inputs[0].stem.upper()}", f"HERIDAL / {inputs[1].stem.upper()}"),
                cell_size=(480, 360),
            )
            gallery.save(args.output_dir / "aerial_samples.jpg", quality=90, optimize=True)
    frames[0].save(
        args.output_dir / "tracking.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(json.dumps({"preview_dir": str(args.output_dir), "image_previews": len(previews), "gif_frames": len(frames)}))


if __name__ == "__main__":
    main()
