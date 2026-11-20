"""Virtual KITTI 2 -> canonical store: segmentation (mapped to Cityscapes names) + depth.

Input: raw/vkitti2/ from scripts/download_vkitti2.sh (Camera_0, six variations, every 5th frame).

- Classes are mapped to Cityscapes names so colours match the Cityscapes evaluator:
  Tree and Vegetation -> vegetation, Van -> car (as in Cityscapes). GuardRail and Misc
  become void, like Cityscapes' ignored guard-rail class. The label space ("vkitti2")
  lists only these 10 classes, so prompts never name a class VKITTI2 does not annotate.
- Depth is kept as the original uint16 centimetres; meta records 100 units per metre. Sky
  (stored by VKITTI2 as 65535 = 655.35 m) is marked invalid (0), as the evaluators mask it anyway.
- Splits: one whole scene (default Scene02, the shortest) is "test", so near-identical neighbouring frames
  cannot leak across splits. Frames are subsampled with `stride`; `val_count` frames of the
  training scenes are held out as "val" for checkpoint selection (they come from the same
  scenes as train, so val tracks fitting rather than generalisation).

    uv run python -m finetune_data.sources.vkitti2 raw/vkitti2 data/finetune_canonical
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from finetune_data import palette, store

SOURCE = "vkitti2"
SKY_DEPTH = 65535  # VKITTI2's depth value for sky / infinity (cm)
VARIATIONS = ["clone", "fog", "morning", "overcast", "rain", "sunset"]
CLASS_MAP = {  # VKITTI2 category -> our class name (None = void)
    "Terrain": "terrain", "Sky": "sky", "Tree": "vegetation", "Vegetation": "vegetation",
    "Building": "building", "Road": "road", "GuardRail": None, "TrafficSign": "traffic sign",
    "TrafficLight": "traffic light", "Pole": "pole", "Misc": None, "Truck": "truck", "Car": "car",
    "Van": "car", "Undefined": None,
}


def read_colors(path: Path) -> dict[tuple[int, int, int], str]:
    colours = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            name, r, g, b = line.split()
            colours[(int(r), int(g), int(b))] = name
    return colours


def colour_lut(colors: dict[tuple[int, int, int], str]) -> dict[int, int]:
    """Packed RGB -> class id in the vkitti2 label space (1-based), void = 0."""
    ids = {name: i for i, (name, _, _) in enumerate(palette.load_palette("vkitti2"), start=1)}
    return {(r << 16) | (g << 8) | b: (ids[CLASS_MAP[name]] if CLASS_MAP[name] else 0)
            for (r, g, b), name in colors.items()}


def to_label(classgt: np.ndarray, lut: dict[int, int]) -> np.ndarray:
    packed = (classgt[..., 0].astype(np.int64) << 16) | (classgt[..., 1].astype(np.int64) << 8) | classgt[..., 2]
    label = np.zeros(packed.shape, dtype=np.uint8)
    for code, class_id in lut.items():
        if class_id:
            label[packed == code] = class_id
    unknown = ~np.isin(packed, list(lut))
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} pixels have colours missing from colors.txt")
    return label


def build(raw_dir: Path, out_root: Path, *, test_scene: str = "Scene02", stride: int = 5,
          variations: list[str] | None = None, val_count: int = 50, seed: int = 0,
          limit: int | None = None) -> list[dict]:
    raw_dir, out = Path(raw_dir), Path(out_root) / SOURCE
    variations = variations or VARIATIONS
    lut = colour_lut(read_colors(raw_dir / "Scene01" / "clone" / "colors.txt"))
    frames = []  # (scene, variation, frame number)
    for scene_dir in sorted(p for p in raw_dir.glob("Scene*") if p.is_dir()):
        for variation in variations:
            rgb_dir = scene_dir / variation / "frames" / "rgb" / "Camera_0"
            for path in sorted(rgb_dir.glob("rgb_*.jpg")):
                number = int(path.stem.split("_")[1])
                if number % stride == 0:
                    frames.append((scene_dir.name, variation, number))
    if not frames:
        raise FileNotFoundError(f"no VKITTI2 frames under {raw_dir}; run scripts/download_vkitti2.sh")
    train_keys = [f for f in frames if f[0] != test_scene]
    val = set(random.Random(seed).sample(train_keys, min(val_count, len(train_keys))))

    rows = []
    for scene, variation, number in frames[:limit] if limit else frames:
        base = raw_dir / scene / variation / "frames"
        sample_id = f"vkitti2_{scene}_{variation}_{number:05d}"
        with Image.open(base / "rgb" / "Camera_0" / f"rgb_{number:05d}.jpg") as image:
            store.save_image(out / "images" / f"{sample_id}.png", np.asarray(image.convert("RGB")))
        with Image.open(base / "classSegmentation" / "Camera_0" / f"classgt_{number:05d}.png") as image:
            store.save_label(out / "seg" / f"{sample_id}.png", to_label(np.asarray(image.convert("RGB")), lut))
        with Image.open(base / "depth" / "Camera_0" / f"depth_{number:05d}.png") as image:
            depth_cm = np.array(image, dtype=np.uint16)
        depth_cm[depth_cm == SKY_DEPTH] = 0  # sky -> invalid
        depth_path = out / "depth" / f"{sample_id}.png"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth_cm).save(depth_path)
        split = "test" if scene == test_scene else ("val" if (scene, variation, number) in val else "train")
        rows.append({"id": sample_id, "source": SOURCE, "split": split, "scene": scene, "variation": variation,
                     "frame": number, "image": f"images/{sample_id}.png", "seg": f"seg/{sample_id}.png",
                     "depth": f"depth/{sample_id}.png"})
    store.write_manifest(out, rows)
    store.write_meta(out, {
        "source": SOURCE, "label_space": "vkitti2", "tasks": ["segmentation", "depth"], "depth_units_per_m": 100.0,
        "test_scene": test_scene, "stride": stride, "variations": variations, "camera": "Camera_0",
        "val_seed": seed, "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "notes": "Classes mapped to Cityscapes names; GuardRail/Misc void; Van -> car; depth in cm, sky = 0 (invalid).",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--test-scene", default="Scene02")
    parser.add_argument("--stride", type=int, default=5, help="keep frames whose number is a multiple of this")
    parser.add_argument("--variations", nargs="+", default=None)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = build(args.raw_dir, args.out_root, test_scene=args.test_scene, stride=args.stride,
                 variations=args.variations, val_count=args.val_count, seed=args.seed, limit=args.limit)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"wrote {len(rows)} images to {args.out_root / SOURCE}: {counts}")


if __name__ == "__main__":
    main()
