"""Drawing helpers for detector output images."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .detector import DetectionResult
from .models import Detection, Track


def draw_detections(image: Image.Image, detections: DetectionResult) -> Image.Image:
    """Return a copy of ``image`` with bounding boxes and labels drawn."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for box, label, score in zip(detections.boxes, detections.labels, detections.scores):
        x1, y1, x2, y2 = box
        caption = f"{label} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 0), width=4)

        left, top, right, bottom = draw.textbbox((x1, y1), caption, font=font)
        text_height = bottom - top
        label_top = max(0, y1 - text_height - 6)
        label_box = (x1, label_top, x1 + (right - left) + 8, label_top + text_height + 6)
        draw.rectangle(label_box, fill=(20, 20, 20))
        draw.text((x1 + 4, label_top + 3), caption, fill=(255, 255, 255), font=font)

    return output


def draw_overlay(
    image: Image.Image,
    *,
    detections: list[Detection] | None = None,
    tracks: list[Track] | None = None,
) -> Image.Image:
    """Draw detections and tracks on a frame for video/pipeline output."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for detection in detections or []:
        _draw_box(draw, detection.box, f"{detection.label} {detection.score:.2f}", (255, 196, 0), font)

    for track in tracks or []:
        caption = f"id {track.track_id} {track.label}"
        _draw_box(draw, track.box, caption, (0, 132, 255), font)

    return output


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    caption: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    left, top, right, bottom = draw.textbbox((x1, y1), caption, font=font)
    text_height = bottom - top
    label_top = max(0, y1 - text_height - 6)
    label_box = (x1, label_top, x1 + (right - left) + 8, label_top + text_height + 6)
    draw.rectangle(label_box, fill=(20, 20, 20))
    draw.text((x1 + 4, label_top + 3), caption, fill=(255, 255, 255), font=font)

