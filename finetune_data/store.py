"""Canonical store: original-resolution images plus annotations in one standard form.

Layout under <root>/<source>/:
    meta.json          source, label space, tasks, units
    manifest.jsonl     one row per image: id, source, split, image, and one path per annotation
    images/<id>.png    RGB input at original resolution
    seg/<id>.png       uint8 class ids in the source's label space (0 = void)
    depth/<id>.png     uint16 depth, meta['depth_units_per_m'] units per metre (default 1000 = mm; 0 = invalid)
    normals/<id>.npy   float16 HxWx3 unit normals in the evaluators' frame (NaN = invalid)

Rendering to (x, t, y) happens later in finetune_data.dataset, so image size, prompts and
target encodings can change without rebuilding the store.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ANNOTATIONS = {"segmentation": "seg", "depth": "depth", "normals": "normals"}


def write_meta(source_dir: Path, meta: dict) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


def read_meta(source_dir: Path) -> dict:
    return json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))


def write_manifest(source_dir: Path, rows: list[dict]) -> None:
    with open(source_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_manifest(source_dir: Path) -> list[dict]:
    with open(source_dir / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def save_label(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(label, dtype=np.uint8)).save(path)


def save_depth(path: Path, depth_m: np.ndarray, units_per_m: float = 1000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    units = np.clip(np.round(np.nan_to_num(depth_m, nan=0.0) * units_per_m), 0, 65535).astype(np.uint16)
    Image.fromarray(units).save(path)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def load_depth(path: Path, units_per_m: float = 1000.0) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float64) / units_per_m


def load_normals(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)
