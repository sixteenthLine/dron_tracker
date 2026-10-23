"""Render clearly labeled dataset-reference previews, never model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from falcon_vision_reconstructed.coco_dataset import CocoDetectionDataset


def render_preview(image: Image.Image, boxes: list[list[float]], *, scene: str) -> Image.Image:
    width = 1280
    scale = width / image.width
    height = round(image.height * scale)
    header, footer = 104, 88
    background, foreground, muted = (17, 27, 31), (243, 248, 249), (174, 192, 198)
    accent = (69, 235, 173)
    canvas = Image.new("RGB", (width, header + height + footer), background)
    canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (0, header))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=27)
    body_font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    box_font = ImageFont.load_default(size=12)
    draw.text((28, 20), "AERIAL OBJECT DETECTION", font=title_font, fill=foreground)
    draw.text((28, 63), f"HERIDAL / {scene.upper()} / CLASS: HUMAN", font=body_font, fill=muted)
    draw.text((900, 23), "REFERENCE ANNOTATIONS", font=body_font, fill=accent)
    draw.text((900, 53), "NOT MODEL OUTPUT", font=body_font, fill=(255, 204, 94))

    frame_boxes = [(left * scale, header + top * scale, right * scale, header + bottom * scale)
                   for left, top, right, bottom in boxes]
    for left, top, right, bottom in frame_boxes:
        draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), outline=background, width=5)
        draw.rectangle((left, top, right, bottom), outline=accent, width=2)

    occupied: list[tuple[float, float, float, float]] = []
    for index, (left, top, right, bottom) in enumerate(frame_boxes, start=1):
        caption = f"{index:02d}"
        label_width = draw.textbbox((0, 0), caption, font=box_font)[2] + 8
        label_height = 20
        candidates = [(left, top - label_height - 2), (left, bottom + 3),
                      (right + 3, top), (left - label_width - 3, top)]
        label = None
        for label_x, label_y in candidates:
            label_x = max(0, min(label_x, width - label_width))
            label_y = max(header, min(label_y, header + height - label_height))
            label = (label_x, label_y, label_x + label_width, label_y + label_height)
            if not any(label[0] < b[2] and label[2] > b[0] and label[1] < b[3] and label[3] > b[1]
                       for b in occupied + frame_boxes):
                break
        assert label is not None
        occupied.append(label)
        draw.rectangle(label, fill=background, outline=accent, width=1)
        draw.text((label[0] + 4, label[1] + 3), caption, font=box_font, fill=foreground)

    footer_top = header + height
    draw.text((28, footer_top + 16), f"{len(boxes):02d} DATASET BOXES  |  COCO REFERENCE LABELS  |  NO CONFIDENCE SCORES",
              font=body_font, fill=foreground)
    draw.text((28, footer_top + 48), "HERIDAL: Bozic-Stulic, Marusic, Gotovac / Roboflow export / CC BY 4.0; upstream CC BY 3.0",
              font=small_font, fill=muted)
    return canvas


def render_previews(annotation_file: Path, image_root: Path, output_dir: Path) -> dict:
    dataset = CocoDetectionDataset(annotation_file, image_root, include_category_ids=[1],
                                   category_id_to_label={1: 1}, validate_images=True)
    records = json.loads(annotation_file.read_text())["images"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "reference_annotations_not_model_output",
                "annotation_sha256": hashlib.sha256(annotation_file.read_bytes()).hexdigest(), "previews": []}
    for index, record in enumerate(records):
        image, target = dataset[index]
        name = f"reference_{Path(record['file_name']).stem}.jpg"
        boxes = target["boxes"].tolist()
        preview = render_preview(image, boxes, scene=Path(record["file_name"]).stem)
        path = output_dir / name
        preview.save(path, quality=93, optimize=True)
        manifest["previews"].append({"file": name, "source_image_id": record["id"],
                                     "source_file": record["file_name"], "boxes": len(boxes),
                                     "dimensions": list(preview.size),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output_dir / "REFERENCE_PREVIEWS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("assets/aerial/reference_annotations.coco.json"))
    parser.add_argument("--image-root", type=Path, default=Path("assets/aerial"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/media"))
    args = parser.parse_args()
    print(json.dumps(render_previews(args.annotations, args.image_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
