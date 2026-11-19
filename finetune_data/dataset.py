"""Union fine-tuning dataset: renders (x, t, y) triplets from the canonical store.

Every choice that the modeling side might want to change is a config value, so nothing
needs rebuilding. Defaults reproduce the preprocessing, prompts and target encodings of
our evaluators. See finetune_data/configs/default.yaml for all options.

    ds = FinetuneDataset(config)            # config: dict or YAML path
    item = ds[i]                            # {"input", "instruction", "target", "task", "source", "id"}
    order = ds.sample_indices(10_000)       # weighted by task/source weights, shuffled
Works directly as a torch Dataset (len / getitem); pass `order` as a sampler.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from finetune_data import encoders, palette, prompts, store
from visionbench.transforms import CoordinateTransform

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"


def load_config(config: dict | str | Path | None = None) -> dict:
    """Default config, deep-updated with `config` (a dict or a YAML path)."""
    merged = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if config is None:
        return merged
    override = config if isinstance(config, dict) else yaml.safe_load(Path(config).read_text(encoding="utf-8"))

    def update(base: dict, new: dict) -> dict:
        for key, value in new.items():
            base[key] = update(base.get(key, {}) or {}, value) if isinstance(value, dict) else value
        return base

    return update(copy.deepcopy(merged), override)


def geometry(size: tuple[int, int], opts: dict) -> CoordinateTransform:
    """Scale (down to max_size / max_area, or to upscale_long_side), then centre-crop to size_multiple."""
    width, height = size
    if opts.get("upscale_long_side"):
        factor = opts["upscale_long_side"] / max(width, height)
    else:
        factor = 1.0
        if opts.get("max_size"):
            factor = min(factor, opts["max_size"][0] / width, opts["max_size"][1] / height)
        if opts.get("max_area"):
            factor = min(factor, math.sqrt(opts["max_area"] / (width * height)))
    scaled = (max(1, round(width * factor)), max(1, round(height * factor)))
    multiple = int(opts.get("size_multiple", 1))
    output = (scaled[0] // multiple * multiple, scaled[1] // multiple * multiple)
    if 0 in output:
        raise ValueError(f"image {size} is too small for size_multiple={multiple}")
    left, top = (scaled[0] - output[0]) // 2, (scaled[1] - output[1]) // 2
    return CoordinateTransform(original_size=(width, height), source_crop=(0, 0, width, height), scaled_size=scaled,
                               output_size=output, scale=(scaled[0] / width, scaled[1] / height),
                               final_crop=(left, top, left + output[0], top + output[1]))


def _resize_array(array: np.ndarray, transform: CoordinateTransform) -> np.ndarray:
    """Nearest-neighbour resample of a per-pixel annotation (labels, depth, normal channels)."""
    if array.ndim == 3:
        return np.stack([_resize_array(array[..., c], transform) for c in range(array.shape[2])], axis=-1)
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(transform.transform_image(image, resample=Image.Resampling.NEAREST))


class FinetuneDataset:
    def __init__(self, config: dict | str | Path | None = None):
        self.config = load_config(config)
        root = Path(self.config["root"])
        enabled_tasks = {t: o for t, o in self.config["tasks"].items() if o.get("enabled", True)}
        splits = set(self.config["splits"])
        self.items: list[tuple[str, dict, str]] = []  # (source, manifest row, task)
        self.sources: dict[str, dict] = {}
        for source, source_opts in self.config["sources"].items():
            if not source_opts.get("enabled", True):
                continue
            source_dir = root / source
            meta = store.read_meta(source_dir)
            self.sources[source] = {"dir": source_dir, "meta": meta, "weight": float(source_opts.get("weight", 1.0)),
                                    "palette": palette.load_palette(meta["label_space"]) if meta.get("label_space") else None}
            for row in store.read_manifest(source_dir):
                if row["split"] not in splits:
                    continue
                for task in enabled_tasks:
                    if row.get(store.ANNOTATIONS[task]):
                        self.items.append((source, row, task))
        if not self.items:
            raise ValueError("no samples match the configured sources, tasks and splits")

    def __len__(self) -> int:
        return len(self.items)

    def counts(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for source, _, task in self.items:
            out[(task, source)] = out.get((task, source), 0) + 1
        return out

    def sample_weights(self) -> np.ndarray:
        """Per-item probabilities. A task gets its share by task weight; within a task each source gets its
        share by source weight; within a (task, source) group items are equally likely, so a small source
        is not drowned out by a large one."""
        counts = self.counts()
        tasks = self.config["tasks"]
        task_sources: dict[str, float] = {}
        for task, source in counts:
            task_sources[task] = task_sources.get(task, 0.0) + self.sources[source]["weight"]
        weights = np.array([float(tasks[task].get("weight", 1.0)) * self.sources[source]["weight"]
                            / task_sources[task] / counts[(task, source)] for source, _, task in self.items])
        return weights / weights.sum()

    def sample_indices(self, num_samples: int | None = None, seed: int | None = None) -> list[int]:
        """Weighted, shuffled indices (with replacement). num_samples defaults to len(self)."""
        rng = np.random.default_rng(self.config.get("seed", 0) if seed is None else seed)
        return rng.choice(len(self), size=num_samples or len(self), p=self.sample_weights()).tolist()

    def task_options(self, task: str) -> dict:
        opts = self.config["tasks"][task]
        return {"size": {**self.config["size"], **(opts.get("size") or {})}, **opts}

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, row, task = self.items[index]
        src = self.sources[source]
        opts = self.task_options(task)
        image = store.load_image(src["dir"] / row["image"])
        transform = geometry(image.size, opts["size"])
        x = transform.transform_image(image, resample=Image.Resampling.LANCZOS)
        annotation = src["dir"] / row[store.ANNOTATIONS[task]]

        if task == "segmentation":
            label = _resize_array(store.load_label(annotation), transform).astype(np.uint8)
            pal = src["palette"]
            label[label > len(pal)] = 0
            listed = (list(range(1, len(pal) + 1)) if opts.get("class_list", "all") == "all"
                      else sorted(int(c) for c in np.unique(label) if c))
            y = encoders.segmentation(label, pal)
            instruction = prompts.segmentation(pal, listed, colour_format=opts.get("colour_format", "name"),
                                               template=opts.get("template") or prompts.SEGMENTATION_TEMPLATE)
        elif task == "depth":
            depth_m = _resize_array(store.load_depth(annotation), transform)
            y = encoders.depth(depth_m, encoding=opts.get("encoding", "disparity"),
                               percentiles=tuple(opts.get("percentiles", (1.0, 99.0))))
            instruction = opts.get("instruction") or prompts.DEPTH
        elif task == "normals":
            xyz = _resize_array(store.load_normals(annotation), transform)
            valid = np.isfinite(xyz).all(axis=-1)
            y = encoders.normals(np.nan_to_num(xyz), valid)
            instruction = opts.get("instruction") or prompts.NORMALS
        else:
            raise ValueError(f"unsupported task {task!r}")
        return {"id": row["id"], "source": source, "task": task, "split": row["split"],
                "input": x, "instruction": instruction, "target": y}
