"""Prepare two small, full-frame HERIDAL examples without changing local originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Heridal.v2-yolo_non_412.coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/aerial"))
    args = parser.parse_args()
    annotation_file = args.dataset / "_annotations.coco.json"
    if not annotation_file.is_file():
        parser.error(f"Local COCO export not found: {annotation_file}")
    data = json.loads(annotation_file.read_text())
    images = {record["id"]: record for record in data["images"]}
    licenses = {record["id"]: record for record in data["licenses"]}
    samples = []
    reference_images = []
    reference_annotations = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, output_name in ((0, "park.jpg"), (100, "terrain.jpg")):
        record = images[image_id]
        license_record = licenses[record["license"]]
        if license_record["url"].rstrip("/") != "https://creativecommons.org/licenses/by/4.0":
            parser.error(f"Unexpected sample license for image {image_id}")
        candidates = [args.dataset / record["file_name"], args.dataset / "train" / record["file_name"]]
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            parser.error(f"Source photo not found: {record['file_name']}")
        destination = args.output_dir / output_name
        with Image.open(source_path) as source:
            original_size = source.size
            sample = ImageOps.contain(source.convert("RGB"), (1600, 1200))
            sample.save(destination, format="JPEG", quality=90, optimize=True)
        scale_x, scale_y = sample.width / original_size[0], sample.height / original_size[1]
        reference_images.append({"id": image_id, "file_name": output_name,
                                 "width": sample.width, "height": sample.height, "license": record["license"]})
        for annotation in data["annotations"]:
            if annotation["image_id"] != image_id:
                continue
            x, y, width, height = annotation["bbox"]
            box = [x * scale_x, y * scale_y, width * scale_x, height * scale_y]
            reference_annotations.append({"id": annotation["id"], "image_id": image_id,
                                          "category_id": annotation["category_id"], "bbox": box,
                                          "area": box[2] * box[3], "iscrowd": annotation.get("iscrowd", 0)})
        samples.append({
            "file": output_name,
            "source_image_id": image_id,
            "source_file_name": record["file_name"],
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "sample_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "original_dimensions": list(original_size),
            "sample_dimensions": list(sample.size),
            "changes": "Full frame resized to fit 1600x1200; JPEG quality 90; no crop; metadata not copied",
            "license": license_record,
        })
    reference_file = args.output_dir / "reference_annotations.coco.json"
    reference_file.write_text(json.dumps({
        "info": {"description": "Reference annotations for two public previews; NOT model predictions",
                 "source_annotation_sha256": hashlib.sha256(annotation_file.read_bytes()).hexdigest(),
                 "coordinate_transform": "Original COCO xywh scaled to each resized sample; no boxes invented"},
        "images": reference_images, "annotations": reference_annotations,
        "categories": data["categories"], "licenses": data["licenses"],
    }, indent=2) + "\n")
    manifest = {
        "dataset": "HERIDAL, local Roboflow v2 yolo_non_412 export",
        "credit": "Dunja Bozic-Stulic, Zeljko Marusic, Sven Gotovac; HERIDAL dataset; Roboflow export",
        "source_url": "https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z",
        "original_dataset_url": "http://ipsar.fesb.unist.hr/HERIDAL%20database.html",
        "export_date": "2022-10-17",
        "annotation_sha256": hashlib.sha256(annotation_file.read_bytes()).hexdigest(),
        "purpose": "Civilian offline still-image portfolio examples; no training or benchmark",
        "reference_annotations": reference_file.name,
        "reference_annotations_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "samples": samples,
    }
    (args.output_dir / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sample_dir": str(args.output_dir), "samples": len(samples)}))


if __name__ == "__main__":
    main()
