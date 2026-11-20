"""Shared helpers for evaluating on the fine-tuning canonical store (finetune_data/).

The in-domain test splits (NYUv2 test, VKITTI2 held-out scene) are read from the same store
the training data comes from, and images are resized with the same `geometry()` the training
renderer uses, so train and test preprocessing cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # plugins are loaded by file path; make finetune_data importable
    sys.path.insert(0, str(REPO_ROOT))

from finetune_data import store  # noqa: E402
from finetune_data.dataset import geometry  # noqa: E402

SIZE_KEYS = ("max_size", "max_area", "upscale_long_side", "size_multiple")


def source_dir(config: dict) -> Path:
    """<store_root>/<source>, store_root relative to the repo (default data/finetune_canonical)."""
    root = Path(config.get("store_root", "data/finetune_canonical"))
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("store_root must be a path inside the repository")
    path = REPO_ROOT / root / config["source"]
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"{path} is not built; see finetune_data/README.md")
    return path


def rows(path: Path, split: str, max_samples: int | None) -> list[dict]:
    selected = [row for row in store.read_manifest(path) if row["split"] == split]
    if not selected:
        raise ValueError(f"no {split!r} rows in {path / 'manifest.jsonl'}")
    return selected[:max_samples] if max_samples else selected


def size_options(config: dict, defaults: dict) -> dict:
    return {key: config.get(key, defaults.get(key)) for key in SIZE_KEYS}


def preprocess(image, opts: dict):
    from PIL import Image
    transform = geometry(image.size, opts)
    resized = transform.transform_image(image, resample=Image.Resampling.LANCZOS)
    parameters = {**opts, "original_size": image.size, "output_size": transform.output_size}
    return resized, transform, parameters
