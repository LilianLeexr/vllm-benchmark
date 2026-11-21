"""Hypersim -> canonical store: surface normals (+ NYU-40 segmentation and depth, opt-in).

Hypersim (Roberts et al., ICCV 2021) is the synthetic indoor set Marigold-normals, StableNormal and
others train their normal estimators on. We follow Marigold's preprocessing, so the stored normals
are in exactly the frame of the Marigold evaluation GT our normal evaluator uses (NYUv2 / iBims /
DIODE): Hypersim's camera-space ``normal_cam``, normalised to unit length, with back-facing normals
(pointing away from the camera) flipped. Images are tone-mapped from the HDR ``color.hdf5`` with
Hypersim's own tone mapper, as Marigold does.

Only the needed members are fetched from each scene zip with HTTP range requests (about 4 MB per
frame instead of ~2 GB per scene). Frames come from Marigold's filtered split lists (public release,
no NaN normals); `train_frames` are spread evenly over all 347 training scenes, and Marigold's
100-image validation subset becomes our "val" split. There is no Hypersim "test" split: normals are
tested on NYUv2 / iBims / DIODE (configs/qwen_normal_*.yaml).

Per frame we also keep planar depth (mm) and NYU-40 semantics (Hypersim's own label space), so the
source can add indoor segmentation / depth data when enabled for those tasks in the config.

    uv run python -m finetune_data.sources.hypersim data/finetune_canonical --train-frames 4000

The build resumes: frames already written are skipped, so rerun it after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from finetune_data import store
from finetune_data.remote_zip import RemoteZip

SOURCE = "hypersim"
SCENES_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
# Pinned so the frame lists and camera parameters never drift.
MARIGOLD_SPLITS = ("https://raw.githubusercontent.com/prs-eth/Marigold/2bfbdeae5d10a50b71f1ba20c865d46e480f6010/"
                   "data_split/hypersim_normals/{name}")
CAMERA_CSV = ("https://raw.githubusercontent.com/apple/ml-hypersim/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/"
              "contrib/mikeroberts3000/metadata_camera_parameters.csv")
WIDTH, HEIGHT = 1024, 768
BACK_FACING_EPS = 1e-3  # Marigold's threshold on n . v


# ----------------------------------------------------------------------------- metadata

def _download_text(url: str, cache: Path) -> str:
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    return cache.read_text(encoding="utf-8")


def parse_split(text: str) -> list[tuple[str, str, int]]:
    """Marigold split lines -> (scene, camera, frame). Accepts 'val/ai_x/rgb_cam_00_fr0031.png ...'."""
    frames = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()[0].split("/")
        scene, name = parts[-2], parts[-1]
        camera, frame = name.removeprefix("rgb_").removesuffix(".png").rsplit("_fr", 1)
        frames.append((scene, camera, int(frame)))
    return frames


def select_evenly(frames: list[tuple[str, str, int]], total: int) -> list[tuple[str, str, int]]:
    """About `total` frames spread over all scenes: an equal quota per scene, evenly spaced in each."""
    by_scene: dict[str, list] = defaultdict(list)
    for frame in frames:
        by_scene[frame[0]].append(frame)
    quota = max(1, math.ceil(total / len(by_scene)))
    chosen = []
    for scene in sorted(by_scene):
        items = sorted(by_scene[scene])
        step = len(items) / min(quota, len(items))
        chosen += [items[int(i * step)] for i in range(min(quota, len(items)))]
    return chosen


def camera_rays(row: dict) -> np.ndarray:
    """Per-pixel camera-space ray directions (H, W, 3) from Hypersim's M_cam_from_uv (handles tilt-shift)."""
    m = np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)])
    u = np.linspace(-1 + 1 / WIDTH, 1 - 1 / WIDTH, WIDTH)
    v = np.linspace(1 - 1 / HEIGHT, -1 + 1 / HEIGHT, HEIGHT)  # image row 0 is v = +1
    uu, vv = np.meshgrid(u, v)
    # ray = M @ [u, v, 1], written element-wise: a large `@` here intermittently segfaults with
    # macOS Accelerate BLAS
    return uu[..., None] * m[:, 0] + vv[..., None] * m[:, 1] + m[:, 2]


# ----------------------------------------------------------------------------- per-frame processing

def tone_map(rgb: np.ndarray, entity_id: np.ndarray) -> np.ndarray:
    """Hypersim's tone mapper (as used by Marigold): 90th-percentile brightness -> 0.8, gamma 1/2.2."""
    gamma = 1.0 / 2.2
    valid = entity_id != -1
    scale = 1.0
    if valid.any():
        brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
        current = np.percentile(brightness[valid], 90)
        scale = 0.0 if current < 1e-4 else 0.8 ** (1.0 / gamma) / current
    return np.clip(np.power(np.maximum(scale * rgb, 0), gamma), 0, 1)


def fix_normals(normal_cam: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Unit length, and flip normals facing away from the camera (Marigold's rule, done in camera space:
    the surface-to-camera direction is -ray)."""
    n = normal_cam / np.maximum(np.linalg.norm(normal_cam, axis=-1, keepdims=True), 1e-12)
    view = -rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    back = (n * view).sum(axis=-1) < -BACK_FACING_EPS
    n[back] *= -1
    return n


def planar_depth(distance: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Hypersim depth_meters is distance to the optical centre; convert to depth along the optical axis."""
    return distance * (-rays[..., 2]) / np.linalg.norm(rays, axis=-1)


_H5_LOCK = threading.Lock()  # h5py on in-memory files is not thread-safe (segfaults); downloads stay parallel


def _h5(z: RemoteZip, name: str) -> np.ndarray:
    data = z.read(name)
    with _H5_LOCK, h5py.File(io.BytesIO(data), "r") as f:
        return np.array(f["dataset"])


def process_frame(z: RemoteZip, scene: str, camera: str, frame: int, rays: np.ndarray, out: Path) -> dict:
    base = f"{scene}/images/scene_{camera}"
    color = _h5(z, f"{base}_final_hdf5/frame.{frame:04d}.color.hdf5").astype(np.float64)
    entity = _h5(z, f"{base}_geometry_hdf5/frame.{frame:04d}.render_entity_id.hdf5").astype(np.int64)
    normal_cam = _h5(z, f"{base}_geometry_hdf5/frame.{frame:04d}.normal_cam.hdf5").astype(np.float64)
    distance = _h5(z, f"{base}_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5").astype(np.float64)
    semantic = _h5(z, f"{base}_geometry_hdf5/frame.{frame:04d}.semantic.hdf5").astype(np.int64)
    if not np.isfinite(normal_cam).all():
        raise ValueError("NaN normals (should have been filtered by the split list)")

    sample_id = f"{SOURCE}_{scene}_{camera}_{frame:04d}"
    store.save_image(out / "images" / f"{sample_id}.png", np.round(tone_map(color, entity) * 255))
    store.save_normals(out / "normals" / f"{sample_id}.npz", fix_normals(normal_cam, rays))
    depth = planar_depth(distance, rays)
    store.save_depth(out / "depth" / f"{sample_id}.png", np.where(np.isfinite(depth) & (depth > 0), depth, 0.0))
    label = np.where((semantic >= 1) & (semantic <= 40), semantic, 0)  # -1 = unlabelled -> void
    store.save_label(out / "seg" / f"{sample_id}.png", label)
    return {"id": sample_id, "source": SOURCE, "scene": scene, "camera": camera, "frame": frame,
            "image": f"images/{sample_id}.png", "normals": f"normals/{sample_id}.npz",
            "depth": f"depth/{sample_id}.png", "seg": f"seg/{sample_id}.png"}


def _row_for(scene: str, camera: str, frame: int) -> dict:
    sample_id = f"{SOURCE}_{scene}_{camera}_{frame:04d}"
    return {"id": sample_id, "source": SOURCE, "scene": scene, "camera": camera, "frame": frame,
            "image": f"images/{sample_id}.png", "normals": f"normals/{sample_id}.npz",
            "depth": f"depth/{sample_id}.png", "seg": f"seg/{sample_id}.png"}


def _done(out: Path, row: dict) -> bool:
    return all((out / row[k]).is_file() for k in ("image", "normals", "depth", "seg"))


def process_scene(scene: str, frames: list[tuple[str, int]], rays_by_scene: dict, out: Path) -> tuple[list, list]:
    todo = [(c, f) for c, f in frames if not _done(out, _row_for(scene, c, f))]
    failures = []
    if todo:
        with RemoteZip(SCENES_URL.format(scene=scene)) as z:
            for camera, frame in todo:
                try:
                    process_frame(z, scene, camera, frame, rays_by_scene[scene], out)
                except Exception as error:  # noqa: BLE001 - report and continue with the other frames
                    failures.append(f"{scene} {camera} {frame}: {error}")
    done = [_row_for(scene, c, f) for c, f in frames if _done(out, _row_for(scene, c, f))]
    return done, failures


# ----------------------------------------------------------------------------- build

def build(out_root: Path, *, train_frames: int = 4000, workers: int = 8, limit_scenes: int | None = None,
          cache_dir: Path = Path("raw/hypersim")) -> list[dict]:
    out = Path(out_root) / SOURCE
    train = select_evenly(parse_split(_download_text(MARIGOLD_SPLITS.format(name="hypersim_filtered_train.txt"),
                                                     cache_dir / "hypersim_filtered_train.txt")), train_frames)
    val = parse_split(_download_text(MARIGOLD_SPLITS.format(name="hypersim_filtered_val_100.txt"),
                                     cache_dir / "hypersim_filtered_val_100.txt"))
    cameras = {row["scene_name"]: row for row in csv.DictReader(io.StringIO(
        _download_text(CAMERA_CSV, cache_dir / "metadata_camera_parameters.csv")))}
    split_of = {**{f: "train" for f in train}, **{f: "val" for f in val}}

    by_scene: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for scene, camera, frame in split_of:
        by_scene[scene].append((camera, frame))
    scenes = sorted(by_scene)[:limit_scenes] if limit_scenes else sorted(by_scene)
    rays_by_scene = {s: camera_rays(cameras[s]) for s in scenes}

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_scene, s, sorted(by_scene[s]), rays_by_scene, out): s for s in scenes}
        for count, future in enumerate(as_completed(futures), start=1):
            done, failed = future.result()
            rows += done
            failures += failed
            print(f"[{count}/{len(scenes)}] {futures[future]}: {len(done)} frames"
                  + (f", {len(failed)} failed" if failed else ""), flush=True)

    for row in rows:
        row["split"] = split_of[(row["scene"], row["camera"], row["frame"])]
    rows.sort(key=lambda r: (r["split"] != "train", r["id"]))
    store.write_manifest(out, rows)
    store.write_meta(out, {
        "source": SOURCE, "label_space": "nyu40", "tasks": ["normals", "segmentation", "depth"],
        "depth_units_per_m": 1000.0, "normals_frame": "Marigold evaluation GT frame (Hypersim normal_cam)",
        "train_frames_requested": train_frames, "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "frame_lists": MARIGOLD_SPLITS.format(name="hypersim_filtered_{train,val_100}.txt"),
        "notes": "Tone-mapped HDR colour; back-facing normals flipped; planar depth in mm; NYU-40 semantics.",
    })
    if failures:
        (out / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{len(failures)} frames failed; see {out / 'failures.txt'} (rerun to retry)", file=sys.stderr)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--train-frames", type=int, default=4000, help="spread evenly over the 347 training scenes")
    parser.add_argument("--workers", type=int, default=8, help="scenes downloaded in parallel")
    parser.add_argument("--limit-scenes", type=int, default=None, help="only the first N scenes (quick check)")
    parser.add_argument("--cache-dir", type=Path, default=Path("raw/hypersim"), help="split lists and camera CSV")
    args = parser.parse_args()
    rows = build(args.out_root, train_frames=args.train_frames, workers=args.workers,
                 limit_scenes=args.limit_scenes, cache_dir=args.cache_dir)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"wrote {len(rows)} frames to {args.out_root / SOURCE}: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
