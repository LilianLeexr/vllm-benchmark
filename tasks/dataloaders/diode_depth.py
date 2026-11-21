"""DIODE val (Marigold packaging) for zero-shot / out-of-domain monocular depth.

Layout under raw/diode_depth/ (scripts/download_diode_depth.sh):
    diode_val_{all,indoor,outdoor}_filename_list.txt   "<rgb> <depth.npy> <mask.npy>" per line
    indoors|outdoor/scene_*/scan_*/<id>.png, <id>_depth.npy (metres), <id>_depth_mask.npy

Pixels outside the validity mask become 0 (invalid). Preprocessing and the prompt are those of
nyuv2_depth.py (RINO prompt; 1024x768 images stay at native size with upscale_long_side 1024), and
tasks/evaluators/nyuv2_depth.py scores it; use Marigold's DIODE range (min 0.6 m, max 350 m) and no
Eigen crop.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.transforms import preprocess_image
from visionbench.types import Sample

_spec = importlib.util.spec_from_file_location("visionbench_task_nyuv2_depth", Path(__file__).with_name("nyuv2_depth.py"))
_nyu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nyu)


class DIODEDepthDataset:
    def __init__(self, root: Path, *, split_file: str, prompt: str, upscale_long_side: int | None, max_size,
                 size_multiple: int, sample_stride: int, max_samples: int | None):
        split = root / split_file
        if not split.is_file():
            raise FileNotFoundError(f"{split} not found; run scripts/download_diode_depth.sh")
        rows = [line.split()[:3] for line in split.read_text().splitlines() if line.strip()][::sample_stride]
        self.rows = rows[:max_samples] if max_samples else rows
        missing = [p for row in self.rows for p in row if not (root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} DIODE files missing, e.g. {root / missing[0]}")
        self.root, self.prompt = root, prompt
        self.upscale_long_side, self.max_size, self.size_multiple = upscale_long_side, max_size, size_multiple

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        rgb_rel, depth_rel, mask_rel = self.rows[index]
        with Image.open(self.root / rgb_rel) as source:
            image = source.convert("RGB")
        depth = np.load(self.root / depth_rel).squeeze().astype(np.float32)
        mask = np.load(self.root / mask_rel).squeeze() > 0
        depth = np.where(mask & np.isfinite(depth), depth, 0.0).astype(np.float32)
        if self.upscale_long_side:
            image, transform, parameters = _nyu._upscale(image, self.upscale_long_side, self.size_multiple)
        else:
            image, transform, parameters = preprocess_image(image, max_size=self.max_size, shrink_mode="scale",
                                                            size_multiple=self.size_multiple)
        return Sample(
            sample_id=Path(rgb_rel).stem, image=image, prompt=self.prompt, ground_truth={"depth": depth},
            metadata={"source_path": rgb_rel, "depth_path": depth_rel, "transform": transform.to_dict(),
                      "preprocessing": parameters},
        )


def load_dataset(config: dict) -> DIODEDepthDataset:
    max_size = config.get("max_size", (1024, 1024))
    return DIODEDepthDataset(
        Path(config["raw_dir"]),
        split_file=config.get("split_file", "diode_val_all_filename_list.txt"),
        prompt=config.get("prompt", _nyu.DEFAULT_PROMPT),
        upscale_long_side=config.get("upscale_long_side", 1024),
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        size_multiple=int(config.get("size_multiple", 32)),
        sample_stride=int(config.get("sample_stride", 1)),
        max_samples=config.get("max_samples"),
    )
