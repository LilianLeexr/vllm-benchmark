"""Fine-tuning data pipeline: targets must decode correctly with our existing evaluators."""

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


NYU_ONLY = {"vkitti2": {"enabled": False}}


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

    ds = FinetuneDataset({"root": str(tmp_path), "sources": {"vkitti2": {"enabled": True, "weight": 1.0}},
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
