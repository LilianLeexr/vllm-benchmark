"""Fine-tuning data pipeline: targets must decode correctly with our existing evaluators."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageColor

from finetune_data import encoders, palette, prompts, store
from finetune_data.dataset import FinetuneDataset, geometry
from visionbench.plugins import load_module

TASKS = Path(__file__).resolve().parents[1] / "tasks"
semseg = load_module(TASKS / "evaluators" / "semseg.py")
depth_eval = load_module(TASKS / "evaluators" / "nyuv2_depth.py")
normal_eval = load_module(TASKS / "evaluators" / "surface_normal.py")
cityscapes = load_module(TASKS / "dataloaders" / "cityscapes.py")


NYU_ONLY = {"vkitti2": {"enabled": False}, "hypersim": {"enabled": False}, "coco_pose": {"enabled": False}}


def _cfg(root: Path, **extra) -> dict:
    return {"root": str(root), "sources": NYU_ONLY, **extra}


def _write_source(root: Path, n: int = 4, size=(640, 480)) -> None:
    """A tiny canonical NYU-like source: segmentation (nyu40) + depth."""
    out = root / "nyuv2"
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        sid = f"nyuv2_{i:04d}"
        label = np.kron(rng.integers(0, 41, (6, 8)), np.ones((80, 80), int))[: size[1], : size[0]]
        yy, xx = np.mgrid[0:size[1], 0:size[0]]
        depth_m = 1.0 + 4.0 * xx / size[0] + 2.0 * yy / size[1]
        store.save_image(out / "images" / f"{sid}.png", rng.integers(0, 255, (size[1], size[0], 3)))
        store.save_label(out / "seg" / f"{sid}.png", label)
        store.save_depth(out / "depth" / f"{sid}.png", depth_m)
        rows.append({"id": sid, "source": "nyuv2", "split": "train" if i < n - 1 else "val",
                     "image": f"images/{sid}.png", "seg": f"seg/{sid}.png", "depth": f"depth/{sid}.png"})
    store.write_manifest(out, rows)
    store.write_meta(out, {"source": "nyuv2", "label_space": "nyu40", "tasks": ["segmentation", "depth"]})


def test_palette_matches_cityscapes_eval_and_names():
    cs = palette.load_palette("cityscapes19")
    assert [(c, n, rgb) for c, n, rgb in cs] == [
        (c, n, tuple(rgb)) for c, (n, rgb) in zip(cityscapes.CLASSES, cityscapes.PROMPT_COLOURS)]
    nyu = palette.load_palette("nyu40")
    assert len({rgb for _, _, rgb in nyu}) == 40 and palette.min_separation("nyu40") > 15
    shared = {c: rgb for c, _, rgb in cs}
    cs_names = {n: rgb for _, n, rgb in cs}  # the Cityscapes eval defines e.g. green, brown, light blue
    for cls, name, rgb in nyu:
        if cls in shared:
            assert rgb == shared[cls]  # same class, same colour across label spaces
        # a colour name always means one RGB: the Cityscapes eval's value if it uses that name, else CSS
        assert rgb == cs_names.get(name, ImageColor.getrgb(name.replace(" ", ""))), name


def test_segmentation_target_roundtrips_through_semseg_decoder():
    pal = palette.load_palette("nyu40")
    label = np.kron(np.arange(41).reshape(1, 41), np.ones((4, 4), int)).astype(np.uint8)
    target = encoders.segmentation(label, pal)
    ids = list(range(1, 41))
    decoded = semseg.decode(target, ids, [rgb for _, _, rgb in pal])
    expected = np.where(label == 0, semseg.IGNORE, label)
    assert np.array_equal(decoded, expected)
    text = prompts.segmentation(pal, [1, 31])
    assert text == "Convert this photo into a color-coded map: the wall purple, the person blue, and everything else black."


def test_depth_target_scores_near_perfect_with_depth_evaluator():
    yy, xx = np.mgrid[0:480, 0:640]
    gt = 1.0 + 4.0 * xx / 640 + 2.0 * yy / 480
    luma = depth_eval.decode_luminance(encoders.depth(gt, percentiles=(0.0, 100.0)))
    scores = depth_eval.score_relative(luma, gt, eigen_crop=False)
    assert scores["abs_rel"] < 0.01 and scores["delta1"] > 0.999


def test_normals_target_roundtrips_with_calibrated_convention():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(32, 32, 3)); xyz[..., 2] = np.abs(xyz[..., 2])
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    rgb = np.asarray(encoders.normals(xyz), dtype=np.float64)
    vectors = 2.0 * rgb / 255.0 - 1.0
    errors, _ = normal_eval.angular_errors(vectors, np.ones((32, 32), bool), xyz, rgb_to_gt=["-r", "+g", "+b"])
    assert errors.max() < 1.0


def test_geometry_matches_eval_preprocessing():
    assert geometry((640, 480), {"max_size": [1024, 1024], "max_area": 524288, "size_multiple": 32}).output_size == (640, 480)
    assert geometry((640, 480), {"upscale_long_side": 1024, "size_multiple": 32}).output_size == (1024, 768)
    assert geometry((2048, 1024), {"max_size": [1024, 1024], "max_area": 524288, "size_multiple": 32}).output_size == (1024, 512)


def test_dataset_renders_triplets_and_balances_sampling(tmp_path):
    _write_source(tmp_path)
    ds = FinetuneDataset(_cfg(tmp_path))
    assert ds.counts() == {("segmentation", "nyuv2"): 3, ("depth", "nyuv2"): 3}
    seg = next(ds[i] for i in range(len(ds)) if ds[i]["task"] == "segmentation")
    assert seg["input"].size == seg["target"].size == (640, 480)
    assert seg["instruction"].count(", the ") == 39  # all 40 classes listed by default
    dep = next(ds[i] for i in range(len(ds)) if ds[i]["task"] == "depth")
    assert dep["input"].size == dep["target"].size == (1024, 768)
    assert np.isclose(ds.sample_weights().sum(), 1.0)

    weighted = FinetuneDataset(_cfg(tmp_path, tasks={"depth": {"weight": 3.0}}))
    tasks = [weighted.items[i][2] for i in weighted.sample_indices(4000, seed=1)]
    assert 0.70 < tasks.count("depth") / len(tasks) < 0.80

    present = FinetuneDataset(_cfg(tmp_path, tasks={"segmentation": {"class_list": "present"},
                                                                "depth": {"enabled": False}}))
    assert all(item[2] == "segmentation" for item in present.items)
    assert present[0]["instruction"].count(", the ") < 39


def test_splits_are_respected(tmp_path):
    _write_source(tmp_path)
    assert {row["split"] for _, row, _ in FinetuneDataset(_cfg(tmp_path)).items} == {"train"}
    assert {row["split"] for _, row, _ in FinetuneDataset(_cfg(tmp_path, splits=["val"])).items} == {"val"}
    with pytest.raises(FileNotFoundError, match="vkitti2"):
        FinetuneDataset({"root": str(tmp_path)})  # enabled by default but not built


def test_vkitti2_classes_map_to_cityscapes_names_and_colours():
    vk = load_module(Path(__file__).resolve().parents[1] / "finetune_data" / "sources" / "vkitti2.py")
    colors = {(90, 200, 255): "Sky", (0, 199, 0): "Tree", (90, 240, 0): "Vegetation", (0, 139, 139): "Van",
              (255, 127, 80): "Car", (250, 100, 255): "GuardRail", (0, 0, 0): "Undefined"}
    lut = vk.colour_lut(colors)
    classgt = np.array([[c for c in colors]], dtype=np.uint8)  # one pixel per colour
    names = {i: c for i, (c, _, _) in enumerate(palette.load_palette("vkitti2"), start=1)}
    got = [names.get(int(i)) for i in vk.to_label(classgt, lut)[0]]
    assert got == ["sky", "vegetation", "vegetation", "car", "car", None, None]
    cs = {c: rgb for c, _, rgb in palette.load_palette("cityscapes19")}
    assert all(cs[c] == rgb for c, _, rgb in palette.load_palette("vkitti2"))  # same colours as the Cityscapes eval
    with pytest.raises(ValueError, match="missing"):
        vk.to_label(np.array([[[1, 2, 3]]], dtype=np.uint8), lut)


def test_depth_units_and_two_source_balance(tmp_path):
    _write_source(tmp_path, n=4)
    out = tmp_path / "vkitti2"
    rows = []
    for i in range(12):  # a larger second source, depth stored in centimetres
        sid = f"vkitti2_{i}"
        store.save_image(out / "images" / f"{sid}.png", np.zeros((375, 1242, 3)))
        store.save_label(out / "seg" / f"{sid}.png", np.ones((375, 1242)))
        store.save_depth(out / "depth" / f"{sid}.png", np.full((375, 1242), 300.0), units_per_m=100.0)
        rows.append({"id": sid, "source": "vkitti2", "split": "train", "image": f"images/{sid}.png",
                     "seg": f"seg/{sid}.png", "depth": f"depth/{sid}.png"})
    store.write_manifest(out, rows)
    store.write_meta(out, {"source": "vkitti2", "label_space": "vkitti2", "depth_units_per_m": 100.0})
    assert np.allclose(store.load_depth(out / "depth" / "vkitti2_0.png", 100.0), 300.0)  # 300 m survives uint16

    ds = FinetuneDataset({"root": str(tmp_path), "sources": {"vkitti2": {"enabled": True, "weight": 1.0}, "hypersim": {"enabled": False}, "coco_pose": {"enabled": False}},
                          "tasks": {"depth": {"enabled": False}}})
    sources = [ds.items[i][0] for i in ds.sample_indices(4000, seed=2)]
    assert 0.45 < sources.count("nyuv2") / len(sources) < 0.55  # 3 vs 12 images, still sampled 50/50
    vk_item = next(ds[i] for i in range(len(ds)) if ds.items[i][0] == "vkitti2")
    assert vk_item["input"].size == (1024, 288)  # long side capped at 1024 like the eval configs, then multiples of 32
    assert vk_item["instruction"].count(", the ") == 9  # 10 VKITTI2 classes, never unannotated Cityscapes ones


def test_vkitti2_sky_depth_becomes_invalid(tmp_path):
    vk = load_module(Path(__file__).resolve().parents[1] / "finetune_data" / "sources" / "vkitti2.py")
    frames = tmp_path / "raw" / "Scene01" / "clone" / "frames"
    (tmp_path / "raw" / "Scene01" / "clone").mkdir(parents=True)
    (tmp_path / "raw" / "Scene01" / "clone" / "colors.txt").write_text("Category r g b\nSky 90 200 255\nRoad 100 60 100\n")
    for sub in ("rgb", "classSegmentation", "depth"):
        (frames / sub / "Camera_0").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(frames / "rgb" / "Camera_0" / "rgb_00000.jpg")
    seg = np.zeros((4, 6, 3), np.uint8); seg[:2] = (90, 200, 255); seg[2:] = (100, 60, 100)
    Image.fromarray(seg).save(frames / "classSegmentation" / "Camera_0" / "classgt_00000.png")
    depth = np.full((4, 6), 1500, np.uint16); depth[:2] = vk.SKY_DEPTH
    Image.fromarray(depth).save(frames / "depth" / "Camera_0" / "depth_00000.png")
    rows = vk.build(tmp_path / "raw", tmp_path / "out", test_scene="Scene99", val_count=0)
    assert [r["split"] for r in rows] == ["train"]
    stored = store.load_depth(tmp_path / "out" / "vkitti2" / rows[0]["depth"], 100.0)
    assert np.all(stored[:2] == 0) and np.allclose(stored[2:], 15.0)  # sky invalid, 15 m kept


def test_indomain_eval_loaders_score_training_targets_perfectly(tmp_path):
    """The in-domain test loaders and the training renderer agree on images, prompts, colours and sizes."""
    _write_source(tmp_path / "store")
    rows = store.read_manifest(tmp_path / "store" / "nyuv2")
    for row in rows:
        row["split"] = "test"
    store.write_manifest(tmp_path / "store" / "nyuv2", rows)
    seg_loader = load_module(TASKS / "dataloaders" / "finetune_semseg.py")
    seg_loader.fs.REPO_ROOT = tmp_path
    train = FinetuneDataset({"root": str(tmp_path / "store"), "splits": ["test"], "sources": NYU_ONLY,
                             "tasks": {"depth": {"enabled": False}}})
    for class_list in ("present", "all"):
        sample = seg_loader.load_dataset({"store_root": "store", "source": "nyuv2", "class_list": class_list})[0]
        item = next(train[i] for i in range(len(train)) if train.items[i][1]["id"] == sample.sample_id)
        assert sample.image.size == item["target"].size
        assert semseg.score(item["target"], sample, {})["pixel_acc"] == 1.0
        if class_list == "all":
            assert sample.prompt == item["instruction"]  # identical to the training instruction

    depth_loader = load_module(TASKS / "dataloaders" / "finetune_depth.py")
    depth_loader.fs.REPO_ROOT = tmp_path
    sample = depth_loader.load_dataset({"store_root": "store", "source": "nyuv2"})[0]
    train = FinetuneDataset({"root": str(tmp_path / "store"), "splits": ["test"], "sources": NYU_ONLY,
                             "tasks": {"segmentation": {"enabled": False}, "depth": {"percentiles": [0, 100]}}})
    item = next(train[i] for i in range(len(train)) if train.items[i][1]["id"] == sample.sample_id)
    assert sample.prompt == item["instruction"] and sample.image.size == item["target"].size == (1024, 768)
    scores = depth_eval.score(item["target"], sample, {"eigen_crop": False})
    assert scores["abs_rel"] < 0.02


def _hypersim():
    return load_module(Path(__file__).resolve().parents[1] / "finetune_data" / "sources" / "hypersim.py")


def test_hypersim_geometry_helpers():
    hs = _hypersim()
    # standard Hypersim camera: 60 deg horizontal FOV, looking down -z
    row = {f"M_cam_from_uv_{i}{j}": v for (i, j), v in np.ndenumerate(np.diag([0.57735, 0.43301, -1.0]))}
    rays = hs.camera_rays(row)
    assert rays.shape == (768, 1024, 3) and np.all(rays[..., 2] == -1.0)
    assert rays[0, 0, 0] < 0 < rays[0, 0, 1]  # top-left pixel: left (-x) and up (+y)
    centre = hs.planar_depth(np.full((768, 1024), 2.0), rays)
    assert np.isclose(centre[384, 512], 2.0, atol=1e-3) and centre[0, 0] < 2.0  # off-axis: distance > depth
    # a wall facing the camera (+z) is kept; a back-facing (-z) one is flipped; lengths become 1
    normals = np.zeros((768, 1024, 3)); normals[..., 2] = 2.0; normals[:10, :, 2] = -1.0
    fixed = hs.fix_normals(normals, rays)
    assert np.allclose(fixed[..., 2], 1.0)
    rgb = np.full((4, 4, 3), 0.5); assert np.all(hs.tone_map(rgb, np.ones((4, 4), int)) <= 1)


def test_hypersim_frame_selection():
    hs = _hypersim()
    text = "\n".join(f"ai_00{s}_001/rgb_cam_00_fr{f:04d}.png x y" for s in range(3) for f in range(100))
    frames = hs.parse_split(text)
    assert frames[0] == ("ai_000_001", "cam_00", 0) and len(frames) == 300
    chosen = hs.select_evenly(frames, 30)
    assert len(chosen) == 30 and {s for s, _, _ in chosen} == {"ai_000_001", "ai_001_001", "ai_002_001"}
    assert [f for s, _, f in chosen if s == "ai_000_001"] == list(range(0, 100, 10))
    assert hs.parse_split("val/ai_052_003/rgb_cam_01_fr0031.png val/x.npy") == [("ai_052_003", "cam_01", 31)]


def test_normals_source_and_per_source_tasks(tmp_path):
    _write_source(tmp_path)
    out = tmp_path / "hypersim"
    xyz = np.zeros((480, 640, 3)); xyz[..., 1] = 1.0; xyz[:240, :, 1] = 0; xyz[:240, :, 2] = 1.0  # floor below, wall above
    rows = []
    for i in range(2):
        sid = f"hypersim_{i}"
        store.save_image(out / "images" / f"{sid}.png", np.zeros((480, 640, 3)))
        store.save_normals(out / "normals" / f"{sid}.npz", xyz)
        store.save_label(out / "seg" / f"{sid}.png", np.ones((480, 640)))
        rows.append({"id": sid, "source": "hypersim", "split": "train", "image": f"images/{sid}.png",
                     "normals": f"normals/{sid}.npz", "seg": f"seg/{sid}.png"})
    store.write_manifest(out, rows)
    store.write_meta(out, {"source": "hypersim", "label_space": "nyu40"})
    assert np.allclose(store.load_normals(out / "normals" / "hypersim_0.npz"), xyz, atol=1e-3)
    ds = FinetuneDataset({"root": str(tmp_path), "sources": NYU_ONLY | {"hypersim": {"tasks": ["normals"]}}})
    assert ds.counts() == {("segmentation", "nyuv2"): 3, ("depth", "nyuv2"): 3, ("normals", "hypersim"): 2}
    item = next(ds[i] for i in range(len(ds)) if ds.items[i][2] == "normals")
    assert item["instruction"] == prompts.NORMALS and item["target"].size == (640, 480)
    target = np.asarray(item["target"], dtype=np.float64)
    errors, _ = normal_eval.angular_errors(2 * target / 255 - 1, np.ones((480, 640), bool), xyz, rgb_to_gt=["-r", "+g", "+b"])
    assert errors.max() < 1.0
    both = FinetuneDataset({"root": str(tmp_path), "sources": NYU_ONLY | {"hypersim": {"tasks": ["normals", "segmentation"]}}})
    assert both.counts()[("segmentation", "hypersim")] == 2


def test_subsets_for_single_task_adapters(tmp_path):
    from finetune_data import export as exporter
    from finetune_data.dataset import subset_config
    _write_source(tmp_path)
    base = {"root": str(tmp_path), "sources": NYU_ONLY}
    depth_only = FinetuneDataset(subset_config(base, tasks=["depth"], sources=["nyuv2"]))
    assert depth_only.counts() == {("depth", "nyuv2"): 3}
    with pytest.raises(ValueError, match="unknown tasks"):
        subset_config(base, tasks=["gesture"])
    assert exporter.groups(base) == [("depth", "nyuv2"), ("segmentation", "nyuv2")]
    n = exporter.export(tmp_path / "out", base, tasks=["segmentation"])
    rows = [json.loads(line) for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    assert n == 3 and {r["task"] for r in rows} == {"segmentation"}
    assert np.isclose(sum(r["sample_weight"] for r in rows), 1.0)


def _upright_person(scale=1.0, dx=0.0, dy=0.0):
    rp = load_module(TASKS / "evaluators" / "_rino_pose.py")
    k = np.zeros((17, 3), np.float32)
    k[:, :2] = rp.TEMPLATE * np.array([300, 600]) * scale + np.array([100 + dx, 80 + dy])
    k[:, 2] = 2
    return rp, k


def test_rino_pose_render_decode_and_metrics():
    rp, k = _upright_person()
    img = rp.render_skeleton(k, rp.CANVAS, joint_radius=6, bone_width=2)
    # an upright frontal person: RINO's template decoder gets the body (its head heuristic is approximate),
    # the colour decoder gets every joint
    assert rp.metrics(rp.decode_pose(img), k, (300.0, 600.0))["c05"] >= 12
    assert rp.metrics(rp.decode_pose_colour(img), k, (300.0, 600.0))["c05"] == 17
    assert rp.average([rp.metrics(k, k, (300.0, 600.0))])["pck@0.05"] == 1.0
    canvas, s, ox, oy = rp.letterbox(Image.new("RGB", (200, 100)))
    assert canvas.size == (512, 768) and np.allclose(rp.unletterbox_kpts(rp.letterbox_kpts(k, s, ox, oy), s, ox, oy), k)


def test_pose_source_renders_rino_canvas(tmp_path):
    rp, k = _upright_person(scale=0.5)
    out = tmp_path / "coco_pose"
    store.save_image(out / "images" / "p0.png", np.zeros((400, 300, 3)))
    store.write_manifest(out, [{"id": "p0", "source": "coco_pose", "split": "train", "image": "images/p0.png",
                                "pose": {"keypoints": k.tolist(), "bbox_wh": [150.0, 300.0]}}])
    store.write_meta(out, {"source": "coco_pose", "tasks": ["pose"]})
    ds = FinetuneDataset({"root": str(tmp_path), "sources": NYU_ONLY | {"nyuv2": {"enabled": False},
                                                                      "coco_pose": {"enabled": True}}})
    item = ds[0]
    assert item["task"] == "pose" and item["input"].size == item["target"].size == (512, 768)
    assert item["instruction"] == rp.PROMPT == prompts.POSE
    _, s, ox, oy = rp.letterbox(Image.new("RGB", (300, 400)))
    pred = rp.unletterbox_kpts(rp.decode_pose_colour(item["target"]), s, ox, oy)
    assert rp.metrics(pred, k, (150.0, 300.0))["c05"] == 17
