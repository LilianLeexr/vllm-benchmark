"""ADE20K val (150 classes) as colour-coded segmentation, following RINO.

Protocol from RINO (arXiv:2607.12450, github.com/yangtiming/RINO,
tasks/semantic_segmentation/evaluate): only the classes present in each image's GT
are listed in the prompt. Palette, prompt templates and the 10 super-category map
are copied from palette_ade20k.py, codec.py and codec_ade10.py.

mode: single
    one fine sample per image (150-class prompt).
palette: rino (default) | named
    named fixes RINO's mismatched colour names; see NAMED_COLOURS.
color_order: frequency (default) | area
    how named colours are handed out among an image's classes: by dataset-wide
    class frequency, or by this image's GT areas (area uses GT; avoid it).
max_pixels: optional cap on processed image area, e.g. 524288 (1024x512) for a
    24 GB GPU; only the two largest val images are affected.
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

# palette: named (our fix). RINO's generated palette pairs names with unrelated RGB
# values (its "orange" is (29, 58, 127)), so the model paints the named colour and is
# decoded against another. Names follow CSS colour values, except "green" is pure
# (0, 255, 0) as in RINO's Cityscapes palette (CSS green is (0, 128, 0)) and
# "dark gray" is (64, 64, 64) (CSS darkgray is lighter than gray). Ten basics
# first, then greedily the farthest remaining colour in CIELAB. Each image
# gives its most frequent classes (see color_order) the first entries; images have
# at most 27 classes.
NAMED_COLOURS = [
    ("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255)), ("yellow", (255, 255, 0)),
    ("cyan", (0, 255, 255)), ("magenta", (255, 0, 255)), ("white", (255, 255, 255)), ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)), ("gray", (128, 128, 128)), ("forest green", (34, 139, 34)),
    ("salmon", (250, 128, 114)), ("royal blue", (65, 105, 225)), ("khaki", (240, 230, 140)),
    ("hot pink", (255, 105, 180)), ("maroon", (128, 0, 0)), ("olive", (128, 128, 0)),
    ("light green", (144, 238, 144)), ("navy", (0, 0, 128)), ("plum", (221, 160, 221)),
    ("sky blue", (135, 206, 235)), ("lime green", (50, 205, 50)), ("crimson", (220, 20, 60)),
    ("chocolate", (210, 105, 30)), ("steel blue", (70, 130, 180)), ("teal", (0, 128, 128)),
    ("tan", (210, 180, 140)), ("pink", (255, 192, 203)), ("violet", (238, 130, 238)),
    ("dark gray", (64, 64, 64)), ("sienna", (160, 82, 45)), ("gold", (255, 215, 0)),
]

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


def load_class_ratios(root: Path) -> list[float]:
    """Pixel ratio of each class over the whole dataset (objectInfo150.txt "Ratio")."""
    ratios = [0.0] * N_CLASSES
    for line in (root / "objectInfo150.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5 and 1 <= int(parts[0]) <= N_CLASSES:
            ratios[int(parts[0]) - 1] = float(parts[1])
    return ratios


def _area_limit(size: tuple[int, int], max_size, max_pixels: int):
    """Tighten max_size so the scaled image has at most max_pixels pixels.

    Images already within the limit keep the original max_size, so only oversized
    ones change (2 of 2,000 ADE20K val images at 1024 px exceed 1024x512 pixels).
    """
    width, height = size
    bound = (max_size, max_size) if isinstance(max_size, int) else (max_size or (width, height))
    factor = min(1.0, bound[0] / width, bound[1] / height)
    if width * height * factor * factor <= max_pixels:
        return max_size
    factor = (max_pixels / (width * height)) ** 0.5
    return (max(1, int(width * factor)), max(1, int(height * factor)))


def read_subset(name: str | None) -> set[str] | None:
    """IDs listed in tasks/subsets/<name> (one per line), or None for the full split."""
    if not name:
        return None
    path = Path(__file__).resolve().parents[1] / "subsets" / name
    if not path.is_file():
        raise FileNotFoundError(f"Subset file not found: {path}; create it with tasks/subsets/make_subsets.py")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def fine_prompt(present: list[int], names: list[str], colours: dict | None = None) -> str:
    colours = colours or {t: PROMPT_COLOURS[t] for t in present}
    parts = [f"the {names[t]} {colours[t][0]}" for t in present]
    return FINE_TEMPLATE.replace("{class_color_list}", ", ".join(parts))


def coarse_prompt(present_cats: list[int]) -> str:
    parts = [f"all {CAT10_GROUP_NAMES[c]} solid {CAT10_PROMPT_COLOURS[c][0]}" for c in present_cats]
    return COARSE_TEMPLATE.replace("{category_color_list}", ", ".join(parts))


class ADE20KDataset:
    def __init__(self, root: Path, *, mode: str, palette: str, color_order: str, max_pixels: int | None, subset: set[str] | None, max_samples: int | None,
                 max_size, shrink_mode: str, size_multiple: int):
        if mode not in {"single", "hierarchical"}:
            raise ValueError("mode must be 'single' or 'hierarchical'")
        image_dir, label_dir = root / "images" / "validation", root / "annotations" / "validation"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Expected images/validation and annotations/validation under {root}")
        self.names = load_class_names(root)
        if palette not in {"rino", "named"}:
            raise ValueError("palette must be 'rino' or 'named'")
        if color_order not in {"frequency", "area"}:
            raise ValueError("color_order must be 'frequency' or 'area'")
        self.mode, self.palette, self.color_order = mode, palette, color_order
        ratios = load_class_ratios(root)
        # Dataset-wide class rank, so colour assignment does not depend on this image's GT areas.
        self.frequency_rank = {c: r for r, c in enumerate(sorted(range(N_CLASSES), key=lambda c: (-ratios[c], c)))}
        self.preprocess = {"max_size": max_size, "shrink_mode": shrink_mode, "size_multiple": size_multiple}
        self.max_pixels = max_pixels
        self.items: list[tuple[Path, Path, list[int]]] = []
        self.by_area: dict[str, list[int]] = {}  # present classes, largest first
        for image_path in sorted(image_dir.glob("*.jpg")):
            if max_samples is not None and len(self.items) >= max_samples:
                break
            if subset is not None and image_path.stem not in subset:
                continue
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.exists():
                continue
            with Image.open(label_path) as label:
                values, areas = np.unique(np.asarray(label, dtype=np.uint8), return_counts=True)
            present = sorted(int(v) - 1 for v in values if 1 <= v <= N_CLASSES)
            self.by_area[image_path.stem] = [int(v) - 1 for _, v in sorted(zip(-areas, values)) if 1 <= v <= N_CLASSES]
            if present:  # RINO skips images with no labelled pixels
                self.items.append((image_path, label_path, present))
        if not self.items:
            raise ValueError(f"No labelled ADE20K validation images found under {root}")
        if subset is not None and max_samples is None and len(self.items) != len(subset):
            missing = sorted(subset - {path.stem for path, _, _ in self.items})
            raise ValueError(f"{len(missing)} subset images missing or unlabelled, e.g. {missing[:3]}")
        self.passes = ["coarse", "fine"] if mode == "hierarchical" else ["fine"]

    def __len__(self) -> int:
        return len(self.items) * len(self.passes)

    def __getitem__(self, index: int) -> Sample:
        image_path, label_path, present = self.items[index // len(self.passes)]
        stage = self.passes[index % len(self.passes)]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        preprocess = dict(self.preprocess)
        if self.max_pixels:
            preprocess["max_size"] = _area_limit(image.size, preprocess["max_size"], self.max_pixels)
        image, transform, parameters = preprocess_image(image, **preprocess)
        if stage == "fine":
            sample_id, class_ids = image_path.stem, present
            if self.palette == "named":
                if len(present) > len(NAMED_COLOURS):
                    raise ValueError(f"{sample_id} has {len(present)} classes; only {len(NAMED_COLOURS)} named colours")
                order = (self.by_area[sample_id] if self.color_order == "area"
                         else sorted(present, key=self.frequency_rank.__getitem__))
                colours = dict(zip(order, NAMED_COLOURS))
            else:
                colours = {t: PROMPT_COLOURS[t] for t in present}
            prompt = fine_prompt(present, self.names, colours)
            colors, lut, n_classes = [colours[t][1] for t in present], FINE_LUT, N_CLASSES
            class_names = self.names
        else:
            class_ids = sorted({TRAINID_TO_10CAT[t] for t in present})
            sample_id, prompt = image_path.stem + COARSE_SUFFIX, coarse_prompt(class_ids)
            colors, lut, n_classes = [CAT10_PROMPT_COLOURS[c][1] for c in class_ids], COARSE_LUT, N_SUPER
            class_names = CAT10_GROUP_NAMES
        return Sample(
            sample_id=sample_id,
            image=image,
            prompt=prompt,
            ground_truth={
                "label_path": str(label_path), "label_lut": lut, "n_classes": n_classes, "class_names": class_names,
                "class_ids": class_ids, "colors": [list(c) for c in colors],
                "stage": stage, "image_id": image_path.stem,
                # Needed by the aggregator to fuse coarse and fine passes.
                "fine_to_coarse": TRAINID_TO_10CAT,
                "coarse_class_names": CAT10_GROUP_NAMES,
            },
            metadata={"source_path": str(image_path), "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> ADE20KDataset:
    max_size = config.get("max_size", (1024, 1024))
    return ADE20KDataset(
        Path(config["raw_dir"]),
        mode=config.get("mode", "hierarchical"),
        palette=config.get("palette", "rino"),
        color_order=config.get("color_order", "frequency"),
        max_pixels=config.get("max_pixels"),
        subset=read_subset(config.get("subset_file")),
        max_samples=config.get("max_samples"),
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 32),
    )
