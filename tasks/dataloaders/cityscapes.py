"""Cityscapes val (19 classes) as colour-coded segmentation.

Prompt template, palette, label mapping and the 7 super-category map are copied
from RINO (arXiv:2607.12450, github.com/yangtiming/RINO,
tasks/semantic_segmentation/evaluate: prompt.txt, palette.py, codec7.py).

class_list: present (RINO)
    list only the classes present in each image's GT.
class_list: all (Vision Banana, arXiv:2604.20329)
    list all 19 classes in every prompt, including absent ones.
mode: single (RINO's Cityscapes setting) | hierarchical
    hierarchical adds a ``<id>__coarse`` 7-group sample per image; the aggregator
    fuses it with the fine pass.

Expected layout under raw/cityscapes/:
    leftImg8bit/val/<city>/<stem>_leftImg8bit.png
    gtFine/val/<city>/<stem>_gtFine_labelIds.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.transforms import preprocess_image
from visionbench.types import Sample

IGNORE = 255
COARSE_SUFFIX = "__coarse"

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
N_CLASSES = len(CLASSES)

# (colour name in the prompt, RGB used for decoding) per train id.
PROMPT_COLOURS = [
    ("red", (255, 0, 0)), ("yellow", (255, 255, 0)), ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)), ("brown", (139, 69, 19)), ("magenta", (255, 0, 255)),
    ("dark yellow", (200, 200, 0)), ("pink", (255, 192, 203)), ("green", (0, 255, 0)),
    ("light green", (144, 238, 144)), ("cyan", (0, 255, 255)), ("blue", (0, 0, 255)),
    ("dark blue", (0, 0, 128)), ("white", (255, 255, 255)), ("gray", (128, 128, 128)),
    ("dark green", (0, 128, 0)), ("dark red", (128, 0, 0)), ("light blue", (128, 128, 255)),
    ("light pink", (255, 182, 193)),
]

FINE_TEMPLATE = "Convert this photo into a color-coded map: {class_color_list}, and everything else black."
COARSE_TEMPLATE = (
    "Turn this image into a flat segmentation mask using only solid colors. "
    "Paint {category_color_list}, and everything else solid black. "
    "Each object must be filled completely as one single-color region from edge to edge "
    "— no internal details should be visible. No textures, no gradients."
)

TRAINID_TO_7CAT = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6]
N_SUPER = 7
CAT7_GROUP_NAMES = [
    "roads and sidewalks", "buildings, walls, and fences", "poles and traffic signs",
    "trees and grass", "sky", "people", "vehicles",
]
CAT7_PROMPT_COLOURS = [
    ("red", (255, 0, 0)), ("orange", (255, 165, 0)), ("magenta", (255, 0, 255)),
    ("green", (0, 255, 0)), ("cyan", (0, 255, 255)), ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
]

# Raw labelId (0..33) -> train id (0..18); everything else is ignored.
_LABEL_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                   23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
FINE_LUT = [_LABEL_TO_TRAIN.get(v, IGNORE) for v in range(256)]
COARSE_LUT = [TRAINID_TO_7CAT[t] if t != IGNORE else IGNORE for t in FINE_LUT]


def read_subset(name: str | None) -> set[str] | None:
    """IDs listed in tasks/subsets/<name> (one per line), or None for the full split."""
    if not name:
        return None
    path = Path(__file__).resolve().parents[1] / "subsets" / name
    if not path.is_file():
        raise FileNotFoundError(f"Subset file not found: {path}; create it with tasks/subsets/make_subsets.py")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def fine_prompt(class_ids: list[int]) -> str:
    parts = [f"the {CLASSES[t]} {PROMPT_COLOURS[t][0]}" for t in class_ids]
    return FINE_TEMPLATE.replace("{class_color_list}", ", ".join(parts))


def coarse_prompt(cat_ids: list[int]) -> str:
    parts = [f"all {CAT7_GROUP_NAMES[c]} solid {CAT7_PROMPT_COLOURS[c][0]}" for c in cat_ids]
    return COARSE_TEMPLATE.replace("{category_color_list}", ", ".join(parts))


class CityscapesDataset:
    def __init__(self, root: Path, *, class_list: str, mode: str, subset: set[str] | None, max_samples: int | None,
                 max_size, shrink_mode: str, size_multiple: int):
        if class_list not in {"present", "all"}:
            raise ValueError("class_list must be 'present' or 'all'")
        if mode not in {"single", "hierarchical"}:
            raise ValueError("mode must be 'single' or 'hierarchical'")
        image_root, label_root = root / "leftImg8bit" / "val", root / "gtFine" / "val"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"Expected leftImg8bit/val and gtFine/val under {root}")
        self.class_list = class_list
        self.preprocess = {"max_size": max_size, "shrink_mode": shrink_mode, "size_multiple": size_multiple}
        lut = np.asarray(FINE_LUT, dtype=np.uint8)
        self.items: list[tuple[Path, Path, list[int]]] = []
        for city in sorted(p for p in image_root.iterdir() if p.is_dir()):
            for image_path in sorted(city.glob("*_leftImg8bit.png")):
                if max_samples is not None and len(self.items) >= max_samples:
                    break
                stem = image_path.name.removesuffix("_leftImg8bit.png")
                if subset is not None and stem not in subset:
                    continue
                label_path = label_root / city.name / f"{stem}_gtFine_labelIds.png"
                if not label_path.exists():
                    continue
                with Image.open(label_path) as label:
                    values = np.unique(lut[np.asarray(label, dtype=np.uint8)])
                present = [int(v) for v in values if v != IGNORE]
                if present:  # RINO skips images with no labelled pixels
                    self.items.append((image_path, label_path, present))
        if not self.items:
            raise ValueError(f"No labelled Cityscapes validation images found under {root}")
        if subset is not None and max_samples is None and len(self.items) != len(subset):
            missing = sorted(subset - {path.name.removesuffix("_leftImg8bit.png") for path, _, _ in self.items})
            raise ValueError(f"{len(missing)} subset images missing or unlabelled, e.g. {missing[:3]}")
        self.passes = ["coarse", "fine"] if mode == "hierarchical" else ["fine"]

    def __len__(self) -> int:
        return len(self.items) * len(self.passes)

    def __getitem__(self, index: int) -> Sample:
        image_path, label_path, present = self.items[index // len(self.passes)]
        stage = self.passes[index % len(self.passes)]
        image_id = image_path.name.removesuffix("_leftImg8bit.png")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image, transform, parameters = preprocess_image(image, **self.preprocess)
        listed = present if self.class_list == "present" else list(range(N_CLASSES))
        if stage == "fine":
            sample_id, class_ids, prompt = image_id, listed, fine_prompt(listed)
            colors, lut, n_classes = [PROMPT_COLOURS[t][1] for t in listed], FINE_LUT, N_CLASSES
            class_names = CLASSES
        else:
            class_ids = sorted({TRAINID_TO_7CAT[t] for t in listed})
            sample_id, prompt = image_id + COARSE_SUFFIX, coarse_prompt(class_ids)
            colors, lut, n_classes = [CAT7_PROMPT_COLOURS[c][1] for c in class_ids], COARSE_LUT, N_SUPER
            class_names = CAT7_GROUP_NAMES
        return Sample(
            sample_id=sample_id,
            image=image,
            prompt=prompt,
            ground_truth={
                "label_path": str(label_path), "label_lut": lut, "n_classes": n_classes, "class_names": class_names,
                "class_ids": class_ids, "colors": [list(c) for c in colors],
                "stage": stage, "image_id": image_id,
                # Needed by the aggregator to fuse coarse and fine passes.
                "fine_to_coarse": TRAINID_TO_7CAT,
                "coarse_class_names": CAT7_GROUP_NAMES,
            },
            metadata={"source_path": str(image_path), "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> CityscapesDataset:
    max_size = config.get("max_size", (1024, 1024))
    return CityscapesDataset(
        Path(config["raw_dir"]),
        class_list=config.get("class_list", "present"),
        mode=config.get("mode", "single"),
        subset=read_subset(config.get("subset_file")),
        max_samples=config.get("max_samples"),
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 32),
    )
