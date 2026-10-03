"""Create the fixed evaluation subsets shared by every model we test.

    uv run python tasks/subsets/make_subsets.py cityscapes   # needs raw/cityscapes
    uv run python tasks/subsets/make_subsets.py ade20k       # needs raw/ade20k

The selection depends only on the official file names and the seed, so anyone with
the dataset regenerates the same list. Commit the .txt files and point
``dataset.params.subset_file`` at them; models evaluated outside VisionBench should
read the same lists.

Cityscapes: stratified by city, proportional to each city's share of val.
ADE20K: uniform sample of labelled val images.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def cityscapes_ids(raw: Path) -> dict[str, list[str]]:
    by_city: dict[str, list[str]] = {}
    for city in sorted(p for p in (raw / "leftImg8bit" / "val").iterdir() if p.is_dir()):
        by_city[city.name] = sorted(p.name.removesuffix("_leftImg8bit.png") for p in city.glob("*_leftImg8bit.png"))
    return by_city


def stratified(groups: dict[str, list[str]], size: int, rng: random.Random) -> list[str]:
    """Largest-remainder allocation of ``size`` across groups, then sample each."""
    total = sum(len(ids) for ids in groups.values())
    quotas = {name: size * len(ids) / total for name, ids in groups.items()}
    counts = {name: int(q) for name, q in quotas.items()}
    for name in sorted(quotas, key=lambda n: quotas[n] - counts[n], reverse=True)[: size - sum(counts.values())]:
        counts[name] += 1
    return sorted(i for name, ids in groups.items() for i in rng.sample(ids, counts[name]))


def ade20k_ids(raw: Path) -> list[str]:
    ids = []
    for label in sorted((raw / "annotations" / "validation").glob("*.png")):
        with Image.open(label) as image:
            if (np.asarray(image) > 0).any():  # skip images with no labelled pixels
                ids.append(label.stem)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["cityscapes", "ade20k"])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    raw = ROOT / "raw" / args.dataset
    if args.dataset == "cityscapes":
        groups = cityscapes_ids(raw)
        chosen = stratified(groups, args.size, rng)
        detail = ", ".join(f"{city}={sum(i.startswith(city) for i in chosen)}" for city in groups)
    else:
        chosen = sorted(rng.sample(ade20k_ids(raw), args.size))
        detail = "uniform"
    path = OUT / f"{args.dataset}_val_{args.size}_seed{args.seed}.txt"
    path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} ids to {path.relative_to(ROOT)} ({detail})")


if __name__ == "__main__":
    main()
