"""Grayscale relative depth on a split of the fine-tuning canonical store (in-domain test).

Same protocol as nyuv2_depth.py (RINO prompt, BT.709 luminance, affine fit in disparity
space, scored by tasks/evaluators/nyuv2_depth.py at original resolution). Ground truth is the
store's metric depth (0 = invalid, e.g. VKITTI2 sky).

    source: vkitti2 (use evaluator max_depth: 80, eigen_crop: false)
    split: test (default) | val
For NYUv2 depth use configs/qwen_nyuv2_depth.yaml instead: it scores the same 654 test images
against the raw sensor depth (Marigold protocol), while the store holds the in-painted depth.
Sizes default to the training renderer's depth geometry (finetune_data/configs/default.yaml).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from visionbench.types import Sample

_spec = importlib.util.spec_from_file_location(
    "visionbench_task_finetune_store", Path(__file__).with_name("_finetune_store.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)  # also puts the repo root on sys.path for finetune_data

from finetune_data import prompts  # noqa: E402

SIZE_DEFAULTS = {"max_size": [1024, 1024], "max_area": None, "upscale_long_side": 1024, "size_multiple": 32}


class FinetuneDepthDataset:
    def __init__(self, config: dict):
        self.dir = fs.source_dir(config)
        self.units_per_m = float(fs.store.read_meta(self.dir).get("depth_units_per_m", 1000.0))
        self.prompt = config.get("prompt") or prompts.DEPTH
        self.size = fs.size_options(config, SIZE_DEFAULTS)
        self.rows = [r for r in fs.rows(self.dir, config.get("split", "test"), config.get("max_samples")) if r.get("depth")]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]
        image, transform, parameters = fs.preprocess(fs.store.load_image(self.dir / row["image"]), self.size)
        depth = fs.store.load_depth(self.dir / row["depth"], self.units_per_m)
        return Sample(
            sample_id=row["id"], image=image, prompt=self.prompt, ground_truth={"depth": depth},
            metadata={"source_path": row["image"], "depth_path": row["depth"], "transform": transform.to_dict(),
                      "preprocessing": parameters},
        )


def load_dataset(config: dict) -> FinetuneDepthDataset:
    return FinetuneDepthDataset(config)
