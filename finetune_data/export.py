"""Write rendered (x, t, y) triplets to disk for training code that prefers files.

    uv run python -m finetune_data.export out_dir [--config my.yaml] [--splits train val] [--limit N]
                                          [--tasks depth] [--sources nyuv2] [--per-group]

Writes out_dir/inputs/<task>/<id>.png, out_dir/targets/<task>/<id>.png and
out_dir/manifest.jsonl (id, source, task, split, instruction, input_path, target_path,
sample_weight). sample_weight reproduces FinetuneDataset.sample_weights() for weighted sampling.

--tasks / --sources keep only those tasks / sources (e.g. the data for one single-task LoRA).
--per-group writes one self-contained folder per (task, source), out_dir/<task>_<source>/,
for training one adapter per task and dataset; the mixed data is the union of these folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune_data.dataset import FinetuneDataset, subset_config


def export(out_dir: Path, config=None, splits: list[str] | None = None, limit: int | None = None,
           tasks: list[str] | None = None, sources: list[str] | None = None) -> int:
    merged = subset_config(config, tasks=tasks, sources=sources)
    if splits:
        merged["splits"] = splits
    ds = FinetuneDataset(merged)
    weights = ds.sample_weights()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
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


def groups(config=None, splits: list[str] | None = None, tasks: list[str] | None = None,
           sources: list[str] | None = None) -> list[tuple[str, str]]:
    """(task, source) pairs that have data under the given selection."""
    merged = subset_config(config, tasks=tasks, sources=sources)
    if splits:
        merged["splits"] = splits
    return sorted(FinetuneDataset(merged).counts())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None, help="keep only these tasks")
    parser.add_argument("--sources", nargs="+", default=None, help="keep only these sources")
    parser.add_argument("--per-group", action="store_true", help="one folder per (task, source)")
    args = parser.parse_args()
    if args.per_group:
        for task, source in groups(args.config, args.splits, args.tasks, args.sources):
            out = args.out_dir / f"{task}_{source}"
            n = export(out, args.config, args.splits, args.limit, tasks=[task], sources=[source])
            print(f"wrote {n} triplets to {out}")
    else:
        n = export(args.out_dir, args.config, args.splits, args.limit, args.tasks, args.sources)
        print(f"wrote {n} triplets to {args.out_dir}")


if __name__ == "__main__":
    main()
