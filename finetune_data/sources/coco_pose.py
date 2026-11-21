"""COCO 2017 person keypoints -> canonical store: human pose estimation (RINO protocol).

Each sample is one person, selected and cropped exactly as RINO's evaluation does (non-crowd, >= 8
visible keypoints, area >= 64^2; GT box expanded by 25% per side; see tasks/evaluators/_rino_pose.py).
The stored image is that crop at original resolution and the manifest row carries the 17 COCO
keypoints in crop pixels plus the tight box size. Letterboxing into the 512x768 canvas and drawing the
OpenPose skeleton target happen when rendering (finetune_data.dataset), so canvas and line sizes stay
configurable.

Training persons come from train2017 only (the test set is val2017, as in RINO); one person per image,
drawn at random with a fixed seed, so `train_count` persons come from `train_count` different images.
`val_count` more persons (from other train2017 images) form the "val" split. Only the selected images
are downloaded (about 160 KB each), plus the keypoint annotations.

    uv run python -m finetune_data.sources.coco_pose raw/coco2017 data/finetune_canonical
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from finetune_data import store
from finetune_data.remote_zip import RemoteZip
from visionbench.plugins import load_module

SOURCE = "coco_pose"
ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"
rino = load_module(Path(__file__).resolve().parents[2] / "tasks" / "evaluators" / "_rino_pose.py")


def annotations(raw_dir: Path) -> dict:
    path = raw_dir / "annotations" / "person_keypoints_train2017.json"
    if not path.is_file():  # pull just this member out of the 253 MB annotation zip
        path.parent.mkdir(parents=True, exist_ok=True)
        with RemoteZip(ANNOTATIONS_ZIP) as z:
            path.write_bytes(z.read("annotations/person_keypoints_train2017.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def select(coco: dict, train_count: int, val_count: int, seed: int) -> list[tuple[str, dict]]:
    """(split, annotation): one random RINO-eligible person per image, images shuffled with `seed`."""
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if rino.keep_annotation(ann):
            by_image.setdefault(int(ann["image_id"]), []).append(ann)
    rng = random.Random(seed)
    image_ids = sorted(by_image)
    rng.shuffle(image_ids)
    chosen = []
    for i, image_id in enumerate(image_ids[: train_count + val_count]):
        people = sorted(by_image[image_id], key=lambda a: a["id"])
        chosen.append(("train" if i < train_count else "val", rng.choice(people)))
    return chosen


def _image(raw_dir: Path, file_name: str) -> Image.Image:
    local = raw_dir / "train2017" / file_name
    if local.is_file():
        data = local.read_bytes()
    else:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(IMAGE_URL.format(file_name=file_name), timeout=60) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 4:
                    raise
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _row(split: str, ann: dict) -> dict:
    sample_id = f"{SOURCE}_{int(ann['image_id']):012d}_{int(ann['id'])}"
    return {"id": sample_id, "source": SOURCE, "split": split, "image_id": int(ann["image_id"]),
            "annotation_id": int(ann["id"]), "image": f"images/{sample_id}.png",
            "pose": None}


def build(raw_dir: Path, out_root: Path, *, train_count: int = 4000, val_count: int = 100, seed: int = 0,
          workers: int = 16) -> list[dict]:
    raw_dir, out = Path(raw_dir), Path(out_root) / SOURCE
    coco = annotations(raw_dir)
    files = {int(im["id"]): im["file_name"] for im in coco["images"]}
    chosen = select(coco, train_count, val_count, seed)

    def one(item):
        split, ann = item
        row = _row(split, ann)
        crop, kpts, bbox_wh, _ = rino.crop_person(_image(raw_dir, files[int(ann["image_id"])]),
                                                   ann["bbox"], ann["keypoints"])
        path = out / row["image"]
        if not path.is_file():
            store.save_image(path, np.asarray(crop))
        row["pose"] = {"keypoints": [[round(float(v), 2) for v in k] for k in kpts],
                       "bbox_wh": [float(bbox_wh[0]), float(bbox_wh[1])]}
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(pool.map(one, chosen), start=1):
            rows.append(row)
            if count % 500 == 0:
                print(f"{count}/{len(chosen)} persons", flush=True)
    store.write_manifest(out, rows)
    store.write_meta(out, {
        "source": SOURCE, "tasks": ["pose"], "protocol": "RINO human pose estimation (top-down crops)",
        "keypoints": "COCO-17 (x, y, v) in crop pixels; v as in COCO (0 = not labelled)",
        "crop_margin": rino.MARGIN, "train_count": train_count, "val_count": val_count, "seed": seed,
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "notes": "train2017 only, one person per image; test = COCO val2017 (configs/coco_pose_estimation*.yaml).",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("raw_dir", type=Path, help="COCO root; annotations are fetched if missing")
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = build(args.raw_dir, args.out_root, train_count=args.train_count, val_count=args.val_count,
                 seed=args.seed, workers=args.workers)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"wrote {len(rows)} persons to {args.out_root / SOURCE}: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
