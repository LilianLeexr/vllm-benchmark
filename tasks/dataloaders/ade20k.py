"""ADE20K val (150 classes) as colour-coded segmentation, following RINO.

Protocol from RINO (arXiv:2607.12450, github.com/yangtiming/RINO,
tasks/semantic_segmentation/evaluate): only the classes present in each image's GT
are listed in the prompt. Palette, prompt templates and the 10 super-category map
are copied from palette_ade20k.py, codec.py and codec_ade10.py.

mode: single
    one fine sample per image (150-class prompt).
mode: hierarchical (RINO --hierarchical, used for its ADE20K numbers)
    two samples per image: ``<id>__coarse`` (10 super-categories) and ``<id>``
    (fine). The aggregator fuses them.

Expected layout under raw/ade20k/ (ADEChallengeData2016 renamed):
    images/validation/*.jpg  annotations/validation/*.png  objectInfo150.txt
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.transforms import preprocess_image
from visionbench.types import Sample

N_CLASSES = 150
N_SUPER = 10
IGNORE = 255
COARSE_SUFFIX = "__coarse"

FINE_TEMPLATE = "Convert this photo into a color-coded map: {class_color_list}, and everything else black."
COARSE_TEMPLATE = (
    "Turn this image into a flat segmentation mask using only solid colors. "
    "Paint {category_color_list}, and everything else solid black. "
    "Each object must be filled completely as one single-color region from edge to edge "
    "— no internal details should be visible. No textures, no gradients."
)

_MODIFIERS = ["", "light", "dark", "bright", "deep", "pale", "vivid", "muted", "soft", "rich"]
_BASE_COLOR_NAMES = [
    "red", "orange", "yellow", "green", "lime", "cyan", "blue", "purple", "pink",
    "brown", "gray", "olive", "teal", "navy", "maroon", "coral", "gold", "violet",
    "salmon", "turquoise",
]

CAT10_GROUP_NAMES = [
    "walls, floors, roads, and stairs",
    "buildings and large structures",
    "sky and water",
    "trees, grass, and plants",
    "furniture such as chairs, tables, and beds",
    "doors, windows, lamps, and signs",
    "people and animals",
    "cars, buses, and other vehicles",
    "appliances and bathroom fixtures",
    "small objects and decorations",
]
CAT10_PROMPT_COLOURS = [
    ("red", (255, 0, 0)), ("orange", (255, 165, 0)), ("cyan", (0, 255, 255)),
    ("green", (0, 255, 0)), ("blue", (0, 0, 255)), ("magenta", (255, 0, 255)),
    ("yellow", (255, 255, 0)), ("purple", (128, 0, 128)), ("teal", (0, 128, 128)),
    ("pink", (255, 192, 203)),
]

_CAT_GROUPS = {
    0: (0, 3, 5, 6, 11, 13, 32, 42, 52, 53, 54, 59, 61, 91, 95, 96, 121, 140),
    1: (1, 25, 48, 51, 79, 84, 88, 101, 114),
    2: (2, 21, 26, 60, 104, 109, 113, 128),
    3: (4, 9, 16, 17, 29, 34, 46, 66, 68, 72, 94),
    4: (7, 10, 15, 19, 23, 24, 28, 30, 31, 33, 35, 39, 44, 45, 56, 57, 62, 64, 69, 70, 73, 75, 77, 97, 99, 110, 117),
    5: (8, 14, 18, 27, 36, 38, 43, 49, 58, 63, 82, 85, 86, 87, 93, 106, 134, 136),
    6: (12, 92, 126),
    7: (20, 76, 80, 83, 90, 102, 103, 105, 116, 127),
    8: (37, 47, 50, 65, 71, 107, 118, 124, 129, 133, 145, 146),
}
TRAINID_TO_10CAT = [9] * N_CLASSES  # group 9 is everything not listed
for _group, _ids in _CAT_GROUPS.items():
    for _i in _ids:
        TRAINID_TO_10CAT[_i] = _group


def build_palette(n: int = N_CLASSES) -> list[tuple[str, tuple[int, int, int]]]:
    """RINO palette_ade20k._build_palette: golden-ratio hues with colour names."""
    palette: list[tuple[str, tuple[int, int, int]]] = []
    used: set[tuple[int, int, int]] = set()
    while len(palette) < n:
        hue = (len(palette) * 0.618033988749895) % 1.0
        sat = 0.45 + 0.45 * ((len(palette) * 7) % 11) / 10
        val = 0.40 + 0.50 * ((len(palette) * 13) % 11) / 10
        rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val))
        if rgb in used:
            continue
        used.add(rgb)
        mod = _MODIFIERS[(len(palette) // len(_BASE_COLOR_NAMES)) % len(_MODIFIERS)]
        base = _BASE_COLOR_NAMES[len(palette) % len(_BASE_COLOR_NAMES)]
        palette.append((f"{mod} {base}" if mod else base, rgb))
    return palette


PROMPT_COLOURS = build_palette()

# Raw annotation value (0 = unlabeled, 1..150) -> eval id.
FINE_LUT = [IGNORE] + list(range(N_CLASSES)) + [IGNORE] * (255 - N_CLASSES)
COARSE_LUT = [IGNORE] + TRAINID_TO_10CAT + [IGNORE] * (255 - N_CLASSES)


def load_class_names(root: Path) -> list[str]:
    """First synonym of each class in objectInfo150.txt, as RINO does."""
    names = [f"class_{i}" for i in range(N_CLASSES)]
    for line in (root / "objectInfo150.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5 and 1 <= int(parts[0]) <= N_CLASSES:
            names[int(parts[0]) - 1] = parts[4].split(",")[0].strip()
    return names


def fine_prompt(present: list[int], names: list[str]) -> str:
    parts = [f"the {names[t]} {PROMPT_COLOURS[t][0]}" for t in present]
    return FINE_TEMPLATE.replace("{class_color_list}", ", ".join(parts))


def coarse_prompt(present_cats: list[int]) -> str:
    parts = [f"all {CAT10_GROUP_NAMES[c]} solid {CAT10_PROMPT_COLOURS[c][0]}" for c in present_cats]
    return COARSE_TEMPLATE.replace("{category_color_list}", ", ".join(parts))


class ADE20KDataset:
    def __init__(self, root: Path, *, mode: str, max_samples: int | None,
                 max_size, shrink_mode: str, size_multiple: int):
        if mode not in {"single", "hierarchical"}:
            raise ValueError("mode must be 'single' or 'hierarchical'")
        image_dir, label_dir = root / "images" / "validation", root / "annotations" / "validation"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Expected images/validation and annotations/validation under {root}")
        self.names = load_class_names(root)
        self.mode = mode
        self.preprocess = {"max_size": max_size, "shrink_mode": shrink_mode, "size_multiple": size_multiple}
        self.items: list[tuple[Path, Path, list[int]]] = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            if max_samples is not None and len(self.items) >= max_samples:
                break
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.exists():
                continue
            with Image.open(label_path) as label:
                values = np.unique(np.asarray(label, dtype=np.uint8))
            present = sorted(int(v) - 1 for v in values if 1 <= v <= N_CLASSES)
            if present:  # RINO skips images with no labelled pixels
                self.items.append((image_path, label_path, present))
        if not self.items:
            raise ValueError(f"No labelled ADE20K validation images found under {root}")
        self.passes = ["coarse", "fine"] if mode == "hierarchical" else ["fine"]

    def __len__(self) -> int:
        return len(self.items) * len(self.passes)

    def __getitem__(self, index: int) -> Sample:
        image_path, label_path, present = self.items[index // len(self.passes)]
        stage = self.passes[index % len(self.passes)]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image, transform, parameters = preprocess_image(image, **self.preprocess)
        if stage == "fine":
            sample_id, class_ids = image_path.stem, present
            prompt = fine_prompt(present, self.names)
            colors, lut, n_classes = [PROMPT_COLOURS[t][1] for t in present], FINE_LUT, N_CLASSES
        else:
            class_ids = sorted({TRAINID_TO_10CAT[t] for t in present})
            sample_id, prompt = image_path.stem + COARSE_SUFFIX, coarse_prompt(class_ids)
            colors, lut, n_classes = [CAT10_PROMPT_COLOURS[c][1] for c in class_ids], COARSE_LUT, N_SUPER
        return Sample(
            sample_id=sample_id,
            image=image,
            prompt=prompt,
            ground_truth={
                "label_path": str(label_path), "label_lut": lut, "n_classes": n_classes,
                "class_ids": class_ids, "colors": [list(c) for c in colors],
                "stage": stage, "image_id": image_path.stem,
                # Needed by the aggregator to fuse coarse and fine passes.
                "fine_to_coarse": TRAINID_TO_10CAT,
            },
            metadata={"source_path": str(image_path), "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> ADE20KDataset:
    max_size = config.get("max_size", (1024, 1024))
    return ADE20KDataset(
        Path(config["raw_dir"]),
        mode=config.get("mode", "hierarchical"),
        max_samples=config.get("max_samples"),
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 32),
    )
