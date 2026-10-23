# Contributing

Run commands from the repository root. Use Python 3.11 and the installation
commands in [README.md](README.md).

```bash
.venv-level-b/bin/python -m pytest
.venv-level-b/bin/aerial-detect run-pipeline --config configs/synthetic_video_demo.toml
```

Keep changes focused and add tests for changed behavior. Synthetic fixtures
must not require a network connection or a model download. ROS2 remains optional;
GOTURN is explicitly a stub. Do not change status labels to "tested" without
recording the corresponding run and its environment.

Data, weights, virtual environments, and generated run outputs stay local. Add
source/license information for any new media to `THIRD_PARTY_NOTICES.md`.
Preview regeneration is documented in [the runbook](docs/RUNBOOK.md).

For the optional offline training module, install `requirements-training.txt`
and run `.venv-level-b/bin/python -m pytest -q tests/test_training.py` from the
root. These tests use synthetic fixtures, not the full export or public weights.
Real training and its positive-only evaluation limitations are documented in
[TRAINING.md](docs/TRAINING.md).
