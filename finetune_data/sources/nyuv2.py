"""NYU Depth V2 labeled set (1,449 images) -> canonical store: segmentation (NYU-40) + depth.

Inputs in <raw_dir> (scripts/download_nyuv2_train.sh):
    nyu_depth_v2_labeled.mat   official MATLAB v7.3 file: images, depths (in-painted, metres)
    labels40.mat               40-class labels for all 1,449 images (Gupta et al. mapping)
    splits.mat                 official 795 train / 654 test indices (1-based)

The 654 test images are the Eigen test split that our depth and normals evaluators use,
so they are marked split="test" and never enter training. A fixed seed holds out
`val_count` train images as split="val" for checkpoint selection.

    uv run python -m finetune_data.sources.nyuv2 raw/nyuv2_train data/finetune
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

from finetune_data import store

SOURCE = "nyuv2"


def build(raw_dir: Path, out_root: Path, *, val_count: int = 50, seed: int = 0, limit: int | None = None) -> list[dict]:
    raw_dir, out = Path(raw_dir), Path(out_root) / SOURCE
    splits = sio.loadmat(raw_dir / "splits.mat")
    train = [int(i) for i in splits["trainNdxs"].ravel()]
    test = [int(i) for i in splits["testNdxs"].ravel()]
    val = set(random.Random(seed).sample(train, val_count))
    split_of = {i: "test" for i in test} | {i: ("val" if i in val else "train") for i in train}
    labels40 = sio.loadmat(raw_dir / "labels40.mat")["labels40"]  # (480, 640, 1449), 0 = unlabeled

    indices = sorted(split_of)[:limit] if limit else sorted(split_of)
    rows = []
    with h5py.File(raw_dir / "nyu_depth_v2_labeled.mat", "r") as mat:
        images, depths = mat["images"], mat["depths"]  # (1449, 3, 640, 480), (1449, 640, 480): MATLAB order
        for index in indices:
            k = index - 1
            sample_id = f"nyuv2_{index:04d}"
            rgb = np.asarray(images[k]).transpose(2, 1, 0)  # -> (480, 640, 3)
            depth_m = np.asarray(depths[k]).T  # -> (480, 640)
            label = labels40[:, :, k]
            assert rgb.shape[:2] == depth_m.shape == label.shape, sample_id
            store.save_image(out / "images" / f"{sample_id}.png", rgb)
            store.save_label(out / "seg" / f"{sample_id}.png", label)
            store.save_depth(out / "depth" / f"{sample_id}.png", depth_m)
            rows.append({"id": sample_id, "source": SOURCE, "split": split_of[index], "nyu_index": index,
                         "image": f"images/{sample_id}.png", "seg": f"seg/{sample_id}.png",
                         "depth": f"depth/{sample_id}.png"})
    store.write_manifest(out, rows)
    store.write_meta(out, {
        "source": SOURCE, "label_space": "nyu40", "tasks": ["segmentation", "depth"], "depth_unit": "mm",
        "splits": {"train": len(train) - val_count, "val": val_count, "test": len(test)}, "val_seed": seed,
        "notes": "Official labeled set; in-painted depths; NYU-40 labels (Gupta et al.); test = Eigen test split.",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N images (debugging)")
    args = parser.parse_args()
    rows = build(args.raw_dir, args.out_root, val_count=args.val_count, seed=args.seed, limit=args.limit)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"wrote {len(rows)} images to {args.out_root / SOURCE}: {counts}")


if __name__ == "__main__":
    main()
