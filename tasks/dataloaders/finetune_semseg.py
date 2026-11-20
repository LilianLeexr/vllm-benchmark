"""Colour-coded segmentation on a split of the fine-tuning canonical store (in-domain test).

Same protocol as cityscapes.py (RINO prompt, nearest-colour decoding, scored at original
resolution by tasks/evaluators/semseg.py), with each source's label space and colours from
finetune_data/palette.json, i.e. exactly the prompts and colours used for training.

    source: nyuv2 (40 NYU classes) | vkitti2 (10 Cityscapes classes)
    split: test (default) | val
    class_list: present (RINO, classes in the GT) | all (Vision Banana, every class)
    colour_format: name (default) | rgb | hex, as in the training config
Sizes default to the training renderer's segmentation geometry (finetune_data/configs/default.yaml).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.types import Sample

_spec = importlib.util.spec_from_file_location(
    "visionbench_task_finetune_store", Path(__file__).with_name("_finetune_store.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)  # also puts the repo root on sys.path for finetune_data

from finetune_data import palette as palettes, prompts  # noqa: E402  (importable after fs)

IGNORE = 255
SIZE_DEFAULTS = {"max_size": [1024, 1024], "max_area": 524288, "upscale_long_side": None, "size_multiple": 32}


class FinetuneSemsegDataset:
    def __init__(self, config: dict):
        self.class_list = config.get("class_list", "present")
        if self.class_list not in {"present", "all"}:
            raise ValueError("class_list must be 'present' or 'all'")
        self.colour_format = config.get("colour_format", "name")
        self.dir = fs.source_dir(config)
        meta = fs.store.read_meta(self.dir)
        self.palette = palettes.load_palette(meta["label_space"])
        self.n_classes = len(self.palette)
        self.class_names = [name for name, _, _ in self.palette]
        # stored ids are 1..N (0 = void) -> eval ids 0..N-1, void ignored
        self.lut = [IGNORE] + list(range(self.n_classes)) + [IGNORE] * (255 - self.n_classes)
        self.size = fs.size_options(config, SIZE_DEFAULTS)
        self.items = []
        for row in fs.rows(self.dir, config.get("split", "test"), config.get("max_samples")):
            label = fs.store.load_label(self.dir / row["seg"])
            present = [int(v) for v in np.unique(label) if 0 < v <= self.n_classes]
            if present:  # RINO skips images with no labelled pixels
                self.items.append((row, present))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        row, present = self.items[index]
        image, transform, parameters = fs.preprocess(fs.store.load_image(self.dir / row["image"]), self.size)
        listed = present if self.class_list == "present" else list(range(1, self.n_classes + 1))
        prompt = prompts.segmentation(self.palette, listed, colour_format=self.colour_format)
        return Sample(
            sample_id=row["id"],
            image=image,
            prompt=prompt,
            ground_truth={
                "label_path": str(self.dir / row["seg"]), "label_lut": self.lut, "n_classes": self.n_classes,
                "class_names": self.class_names, "class_ids": [c - 1 for c in listed],
                "colors": [list(self.palette[c - 1][2]) for c in listed],
                "stage": "fine", "image_id": row["id"],
                "fine_to_coarse": list(range(self.n_classes)),  # no coarse pass; identity keeps the aggregator happy
            },
            metadata={"source_path": row["image"], "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> FinetuneSemsegDataset:
    return FinetuneSemsegDataset(config)
