"""Write rendered (x, t, y) triplets to disk for training code that prefers files.

    uv run python -m finetune_data.export out_dir [--config my.yaml] [--splits train val] [--limit N]

Writes out_dir/inputs/<task>/<id>.png, out_dir/targets/<task>/<id>.png and
out_dir/manifest.jsonl (id, source, task, split, instruction, input_path, target_path,
sample_weight). sample_weight reproduces FinetuneDataset.sample_weights() for weighted sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune_data.dataset import FinetuneDataset


def export(out_dir: Path, config=None, splits: list[str] | None = None, limit: int | None = None) -> int:
    overrides = dict(config or {}) if not isinstance(config, (str, Path)) else config
    if splits:
        overrides = {**(overrides if isinstance(overrides, dict) else {}), "splits": splits}
    ds = FinetuneDataset(overrides or None)
    weights = ds.sample_weights()
    out_dir = Path(out_dir)
    count = 0
    with open(out_dir.mkdir(parents=True, exist_ok=True) or out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for index in range(len(ds) if limit is None else min(limit, len(ds))):
            item = ds[index]
            rel_x = Path("inputs") / item["task"] / f"{item['id']}.png"
            rel_y = Path("targets") / item["task"] / f"{item['id']}.png"
            for rel, image in ((rel_x, item["input"]), (rel_y, item["target"])):
                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                image.save(out_dir / rel)
            f.write(json.dumps({"id": item["id"], "source": item["source"], "task": item["task"],
                                "split": item["split"], "instruction": item["instruction"],
                                "input_path": str(rel_x), "target_path": str(rel_y),
                                "sample_weight": float(weights[index])}) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(f"wrote {export(args.out_dir, args.config, args.splits, args.limit)} triplets to {args.out_dir}")


if __name__ == "__main__":
    main()
