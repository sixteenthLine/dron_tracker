"""Check Git publication contents and optionally smoke-test an isolated copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Test a copy containing only publishable files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    listing = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    files = sorted({Path(os.fsdecode(name)) for name in listing.split(b"\0") if name})
    errors: list[str] = []
    blocked_roots = {"outputs", "data", "datasets", "checkpoints", ".pytest_cache", "__pycache__"}
    blocked_suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".caffemodel", ".npy", ".pyc", ".pem", ".key"}
    for file in files:
        path = root / file
        if (file.parts[0] in blocked_roots or file.parts[0].startswith((".venv", "Heridal"))
                or file.suffix in blocked_suffixes or file.name == ".DS_Store"
                or file.name == ".env" or file.name.startswith(".env.") and file.name != ".env.example"
                or file.as_posix() in {"assets/sample_civilian_dog.jpg", "assets/sample_public_dog.jpg"}):
            errors.append(f"Local-only file would be published: {file}")
        if path.is_symlink() or not path.is_file():
            errors.append(f"Missing file or unsupported symlink: {file}")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"Unexpected large publication file: {file}")
        if file.suffix == ".md":
            content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
            for link in re.findall(r"\]\(([^)]+)\)", content):
                target = link.split("#", 1)[0]
                if target and not target.startswith(("https:", "http:", "mailto:")):
                    resolved = (path.parent / target).resolve()
                    if not resolved.is_relative_to(root) or not resolved.exists():
                        errors.append(f"Broken/outside local link in {file}: {target}")
    manifest_path = root / "assets/aerial/PROVENANCE.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for sample in manifest["samples"]:
            path = (manifest_path.parent / sample["file"]).resolve()
            if not path.is_relative_to(manifest_path.parent) or not path.is_file():
                errors.append(f"Missing or invalid aerial sample: {sample['file']}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != sample["sample_sha256"]:
                errors.append(f"Aerial sample hash mismatch: {sample['file']}")
        if "reference_annotations" in manifest:
            reference = manifest_path.parent / manifest["reference_annotations"]
            if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != manifest["reference_annotations_sha256"]:
                errors.append("Reference annotation hash mismatch or missing file")
    preview_manifest = root / "docs/media/REFERENCE_PREVIEWS.json"
    if preview_manifest.is_file():
        previews = json.loads(preview_manifest.read_text())
        if previews["kind"] != "reference_annotations_not_model_output":
            errors.append("Reference previews must not be labeled as model output")
        reference = root / "assets/aerial/reference_annotations.coco.json"
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != previews["annotation_sha256"]:
            errors.append("Reference preview annotation source mismatch")
        for entry in previews["previews"]:
            path = (preview_manifest.parent / entry["file"]).resolve()
            if not path.is_relative_to(preview_manifest.parent) or not path.is_file():
                errors.append(f"Missing reference preview: {entry['file']}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                errors.append(f"Reference preview hash mismatch: {entry['file']}")
    if errors:
        raise SystemExit("\n".join(errors))

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="aerial-publication-") as directory:
            clean = Path(directory)
            for file in files:
                destination = clean / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / file, destination)
            env = dict(os.environ, PYTHONPATH=str(clean / "src"), PYTHONDONTWRITEBYTECODE="1",
                       CUDA_VISIBLE_DEVICES="", TORCH_HOME=str(clean / ".torch"))
            commands = [
                ["-c", "from pathlib import Path; import falcon_vision_reconstructed as f; assert Path(f.__file__).resolve().is_relative_to(Path.cwd())"],
                ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
                ["-m", "falcon_vision_reconstructed.cli", "create-demo-video", "--frames", "24", "--fps", "12"],
                ["-m", "falcon_vision_reconstructed.cli", "run-pipeline", "--config", "configs/synthetic_video_demo.toml"],
                ["-m", "falcon_vision_reconstructed.cli", "infer-video", "--input", "outputs/synthetic_civilian_input.mp4", "--output", "outputs/synthetic_civilian_infer_video.mp4", "--detector", "mock", "--tracker", "iou", "--max-frames", "8", "--fps", "12"],
                ["-m", "falcon_vision_reconstructed.cli", "validate"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, *command], cwd=clean, env=env,
                                        capture_output=True, text=True)
                if result.returncode:
                    raise SystemExit(result.stdout + result.stderr)
            videos = list((clean / "outputs").glob("*.mp4"))
            if len(videos) != 3 or any(path.stat().st_size == 0 for path in videos):
                raise SystemExit("The clean-copy demo did not produce three nonempty videos")
    print(json.dumps({"files": len(files), "bytes": sum((root / file).stat().st_size for file in files),
                      "links": "passed", "local_file_exclusions": "passed",
                      "clean_copy_smoke": "passed" if args.smoke else "not_run"}, sort_keys=True))


if __name__ == "__main__":
    main()
