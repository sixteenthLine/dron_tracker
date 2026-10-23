# GitHub Publication

Public name: **Aerial Object Detection**. Repository: `aerial-object-detection`.
Suggested description: `Offline aerial object detection with COCO utilities,
tiled fine-tuning, reproducible demos, and documented evaluation limits.`
Topics: `computer-vision`, `pytorch`, `torchvision`, `object-detection`, `aerial-imagery`.

This is independently named new work, not an official continuation of the former
team. "Moving drone" is not advertised as a verified capability: real aerial
video and camera-motion handling have not been validated.

## Working Directory

The current local folder is still `falcon-vision-reconstructed`; renaming it
would invalidate existing virtual-environment shebangs and saved local links.
The GitHub repository name does not need to match this local folder.

From the parent workspace, first run:

```bash
cd falcon-vision-reconstructed
```

Every command below runs **inside that repository root**. Future clones will
normally be named `aerial-object-detection` instead.

## Check Before Staging

```bash
.venv-level-b/bin/python -m pytest -q
.venv-level-b/bin/python scripts/verify_publication.py --smoke
git status --short
git check-ignore .venv-level-b/ Heridal.v2-yolo_non_412.coco/ outputs/
git add --dry-run .
```

Do not upload the full dataset, weights, environments, old local photos, or
historical binaries. Reference previews are explicitly annotations, not successful
model predictions. The ignore rules and publication check exclude local artifacts.

## First Commit

Git is already initialized locally; no commit or remote currently exists.
Review the staged filenames and diff before committing:

```bash
git branch -M main
git add .
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
git commit -m "Add reproducible aerial object detection experiments"
```

If Git requests an identity, configure your chosen name and GitHub verified or
privacy email locally, then retry the commit. Do not paste access tokens into
files or the repository URL.

## Push to GitHub

Create an **empty** GitHub repository named `aerial-object-detection`, without
an initial README, license, or gitignore. Replace `YOUR_GITHUB_USERNAME` below
with your account or organization, and authenticate using your usual Git setup.

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/aerial-object-detection.git
git remote -v
git push -u origin main
```

If `origin` is already configured, inspect `git remote -v` and use
`git remote set-url origin ...` with the intended URL instead of adding it again.
No force push is needed. These commit/remote/push commands are instructions;
they were **not executed** during preparation.

## Later Updates

From the repository root, after reviewing changes:

```bash
git status --short
.venv-level-b/bin/python scripts/verify_publication.py
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Update aerial detection demo"
git push
```

Review the staged filenames before committing. GitHub-hosted CI remains
pending until publication; no passing badge is claimed beforehand.

## Colab Bundle

From the root:

```bash
.venv-level-b/bin/python scripts/build_training_bundle.py
```

Output: `outputs/aerial_detection_training.zip`, rooted at
`aerial-object-detection/`. It contains source and small public samples, not the
dataset or checkpoints. Open `notebooks/Aerial_Detection_Training.ipynb` in Colab.
Uploading the ZIP does not itself authorize Drive or start GPU training.
