"""Package publishable project files for upload to Colab, without datasets/weights."""

from pathlib import Path
import os
import subprocess
import sys
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/verify_publication.py"], cwd=root, check=True)
    listing = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    files = sorted({Path(os.fsdecode(name)) for name in listing.split(b"\0") if name})
    destination = root / "outputs/aerial_detection_training.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(root / path, Path("aerial-object-detection") / path)
    print(f"Created outputs/aerial_detection_training.zip: {len(files)} project files, no full dataset or weights")


if __name__ == "__main__":
    main()
