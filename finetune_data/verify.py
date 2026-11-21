"""Check that a locally built canonical store matches ours, pixel for pixel.

    uv run python -m finetune_data.verify data/finetune_canonical            # compare with checksums/
    uv run python -m finetune_data.verify data/finetune_canonical --write    # (maintainers) record them

Hashes decoded content, not file bytes: images / labels / depth by their pixel arrays, normals by
their float16 arrays, manifest rows and meta as canonical JSON. So a different PNG encoder or zlib
version gives the same result, while any change in the data itself is reported per file.
Checksums live in finetune_data/checksums/<source>.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from finetune_data import store

CHECKSUMS = Path(__file__).with_name("checksums")
FILE_KEYS = ("image", "seg", "depth", "normals")


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}{array.shape}".encode())
    h.update(array.tobytes())
    return h.hexdigest()[:32]


def content_hash(path: Path) -> str:
    if path.suffix in (".npz", ".npy"):
        return _array_hash(store.load_normals(path).astype(np.float16))
    with Image.open(path) as image:
        return _array_hash(np.asarray(image))


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:32]


def source_checksums(source_dir: Path, workers: int = 8) -> dict:
    rows = store.read_manifest(source_dir)
    meta = store.read_meta(source_dir)
    files = sorted({row[k] for row in rows for k in FILE_KEYS if isinstance(row.get(k), str)})
    with ThreadPoolExecutor(workers) as pool:
        hashes = dict(zip(files, pool.map(lambda f: content_hash(source_dir / f), files)))
    return {"meta": _json_hash({k: v for k, v in meta.items()}), "rows": {r["id"]: _json_hash(r) for r in rows},
            "files": hashes}


def compare(expected: dict, actual: dict) -> list[str]:
    problems = []
    if expected["meta"] != actual["meta"]:
        problems.append("meta.json differs")
    for kind in ("rows", "files"):
        exp, act = expected[kind], actual[kind]
        missing, extra = sorted(set(exp) - set(act)), sorted(set(act) - set(exp))
        changed = sorted(k for k in set(exp) & set(act) if exp[k] != act[k])
        for label, items in (("missing", missing), ("unexpected", extra), ("different", changed)):
            if items:
                problems.append(f"{len(items)} {label} {kind}, e.g. {items[:3]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources", nargs="+", default=None, help="default: every source with a checksum file")
    parser.add_argument("--write", action="store_true", help="record checksums of this store instead of comparing")
    args = parser.parse_args()
    if args.write:
        sources = args.sources or sorted(p.name for p in args.root.iterdir() if (p / "meta.json").is_file())
    else:
        sources = args.sources or sorted(p.name.removesuffix(".json.gz") for p in CHECKSUMS.glob("*.json.gz"))
    failed = False
    for source in sources:
        actual = source_checksums(args.root / source)
        target = CHECKSUMS / f"{source}.json.gz"
        if args.write:
            CHECKSUMS.mkdir(exist_ok=True)
            with gzip.open(target, "wt", encoding="utf-8") as f:
                json.dump(actual, f, sort_keys=True)
            print(f"{source}: recorded {len(actual['rows'])} rows, {len(actual['files'])} files")
            continue
        with gzip.open(target, "rt", encoding="utf-8") as f:
            problems = compare(json.load(f), actual)
        failed |= bool(problems)
        print(f"{source}: " + ("OK, identical" if not problems else "MISMATCH\n  " + "\n  ".join(problems)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
