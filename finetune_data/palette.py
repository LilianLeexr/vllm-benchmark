"""Global class-to-colour table for segmentation targets.

Rules:
- The same class name gets the same colour in every label space.
- Cityscapes classes keep the exact colours of tasks/dataloaders/cityscapes.py, so a
  fine-tuned model is scored by the existing Cityscapes "all classes" evaluation.
- Colours only need to be distinct within one label space, because a prompt never
  mixes label spaces. New classes get the named colour farthest (CIELAB) from the
  colours already used in their label space.
- A colour name always means one RGB value: the Cityscapes evaluator's value when it
  uses that name (e.g. "green" = (0, 255, 0), "brown" = (139, 69, 19)), otherwise CSS.
- Known issue kept for compatibility: the Cityscapes evaluator's "pink" (traffic sign)
  and "light pink" (bicycle) are nearly identical (CIELAB distance 4.9).

`python -m finetune_data.palette` rewrites finetune_data/palette.json; datasets read
the frozen JSON so targets never change silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import ImageColor

PALETTE_PATH = Path(__file__).with_name("palette.json")
REPO = Path(__file__).resolve().parents[1]

# Candidate pool: (prompt name, CSS key). Black is reserved for "everything else".
_CSS_POOL = [
    ("red", "red"), ("green", "lime"), ("blue", "blue"), ("yellow", "yellow"), ("cyan", "cyan"),
    ("magenta", "magenta"), ("orange", "orange"), ("purple", "purple"), ("pink", "pink"),
    ("brown", "saddlebrown"), ("gray", "gray"), ("white", "white"), ("navy", "navy"),
    ("maroon", "maroon"), ("olive", "olive"), ("teal", "teal"), ("gold", "gold"), ("silver", "silver"),
    ("beige", "beige"), ("coral", "coral"), ("salmon", "salmon"), ("khaki", "khaki"), ("tan", "tan"),
    ("orchid", "orchid"), ("plum", "plum"), ("violet", "violet"), ("indigo", "indigo"),
    ("turquoise", "turquoise"), ("crimson", "crimson"), ("chocolate", "chocolate"), ("sienna", "sienna"),
    ("tomato", "tomato"), ("orange red", "orangered"), ("dark orange", "darkorange"),
    ("goldenrod", "goldenrod"), ("yellow green", "yellowgreen"), ("olive drab", "olivedrab"),
    ("forest green", "forestgreen"), ("dark green", "darkgreen"), ("sea green", "seagreen"),
    ("spring green", "springgreen"), ("light green", "lightgreen"), ("aquamarine", "aquamarine"),
    ("dark cyan", "darkcyan"), ("steel blue", "steelblue"), ("sky blue", "skyblue"),
    ("deep sky blue", "deepskyblue"), ("dodger blue", "dodgerblue"), ("royal blue", "royalblue"),
    ("cornflower blue", "cornflowerblue"), ("slate blue", "slateblue"), ("medium purple", "mediumpurple"),
    ("dark violet", "darkviolet"), ("dark magenta", "darkmagenta"), ("deep pink", "deeppink"),
    ("hot pink", "hotpink"), ("light pink", "lightpink"), ("rosy brown", "rosybrown"), ("peru", "peru"),
    ("sandy brown", "sandybrown"), ("wheat", "wheat"), ("lavender", "lavender"), ("dark red", "darkred"),
    ("firebrick", "firebrick"), ("light coral", "lightcoral"), ("dark khaki", "darkkhaki"),
    ("slate gray", "slategray"), ("dark slate gray", "darkslategray"), ("light gray", "lightgray"),
    ("midnight blue", "midnightblue"), ("burlywood", "burlywood"), ("chartreuse", "chartreuse"),
    ("light sea green", "lightseagreen"), ("dark turquoise", "darkturquoise"), ("thistle", "thistle"),
    ("dark orchid", "darkorchid"), ("blue violet", "blueviolet"), ("light salmon", "lightsalmon"),
]
POOL = [(name, ImageColor.getrgb(css)) for name, css in _CSS_POOL]

NYU40_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor mat", "clothes", "ceiling", "books", "refrigerator", "television", "paper", "towel",
    "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "other structure", "other furniture", "other prop",
]


# Virtual KITTI 2 classes after mapping to Cityscapes names (finetune_data.sources.vkitti2).
VKITTI2_CLASSES = ["road", "building", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
                   "sky", "car", "truck"]


def _lab(rgb) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = x @ np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]]).T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def cityscapes_palette() -> dict[str, tuple[str, tuple[int, int, int]]]:
    """Class -> (colour name, RGB) exactly as the Cityscapes evaluator prompts them."""
    from visionbench.plugins import load_module

    cs = load_module(REPO / "tasks" / "dataloaders" / "cityscapes.py")
    return {cls: (name, tuple(rgb)) for cls, (name, rgb) in zip(cs.CLASSES, cs.PROMPT_COLOURS)}


def assign_label_space(classes: list[str], fixed: dict[str, tuple[str, tuple[int, int, int]]]):
    """Colours for one label space: shared classes reuse `fixed`, others take the
    pool colour farthest from every colour already used here (and from black)."""
    chosen: dict[str, tuple[str, tuple[int, int, int]]] = {c: fixed[c] for c in classes if c in fixed}
    used = [_lab((0, 0, 0))] + [_lab(rgb) for _, rgb in chosen.values()]
    taken = {rgb for _, rgb in chosen.values()}
    for cls in classes:
        if cls in chosen:
            continue
        options = [(n, rgb) for n, rgb in POOL if rgb not in taken]
        name, rgb = max(options, key=lambda o: min(np.linalg.norm(_lab(o[1]) - u) for u in used))
        chosen[cls] = (name, rgb); used.append(_lab(rgb)); taken.add(rgb)
    return chosen


def build_palette() -> dict:
    """{label_space: [[class, colour name, [r, g, b]], ...]} in each space's class order."""
    global POOL
    cs = cityscapes_palette()
    cs_names = {name: rgb for name, rgb in cs.values()}
    POOL = [(name, cs_names.get(name, rgb)) for name, rgb in POOL]  # one RGB per colour name
    global_colours = dict(cs)
    spaces = {"cityscapes19": list(cs), "nyu40": NYU40_CLASSES, "vkitti2": VKITTI2_CLASSES}
    out = {}
    for space, classes in spaces.items():
        colours = assign_label_space(classes, global_colours)
        global_colours.update(colours)
        out[space] = [[c, colours[c][0], list(colours[c][1])] for c in classes]
    return out


def load_palette(space: str) -> list[tuple[str, str, tuple[int, int, int]]]:
    """Frozen (class, colour name, RGB) list for a label space, in class-id order (id = index + 1)."""
    data = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    return [(c, n, tuple(rgb)) for c, n, rgb in data[space]]


def min_separation(space: str) -> float:
    """Smallest CIELAB distance between two colours (or a colour and black) in a space."""
    labs = [_lab((0, 0, 0))] + [_lab(rgb) for _, _, rgb in load_palette(space)]
    return min(np.linalg.norm(a - b) for i, a in enumerate(labs) for b in labs[i + 1:])


if __name__ == "__main__":
    PALETTE_PATH.write_text(json.dumps(build_palette(), indent=1) + "\n", encoding="utf-8")
    for space in json.loads(PALETTE_PATH.read_text()):
        print(f"{space}: min CIELAB separation {min_separation(space):.1f}")
