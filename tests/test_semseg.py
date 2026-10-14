from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageColor

from visionbench.plugins import load_module
from visionbench.transforms import CoordinateTransform

TASKS = Path(__file__).resolve().parents[1] / "tasks"
ade = load_module(TASKS / "dataloaders" / "ade20k.py")
cityscapes = load_module(TASKS / "dataloaders" / "cityscapes.py")
evaluator = load_module(TASKS / "evaluators" / "semseg.py")
aggregator = load_module(TASKS / "aggregators" / "semseg.py")


class _Reference:
    """Stand-in for the runner's lazy sample reference."""

    def __init__(self, sample):
        self.sample = sample

    def load(self):
        return self.sample

    def __getattr__(self, name):
        return getattr(self.sample, name)


def _write_ade(root: Path, images: dict[str, np.ndarray]) -> None:
    """ADE20K layout with raw labels (0 = unlabeled, 1..150 = classes)."""
    root.mkdir(parents=True, exist_ok=True)
    lines = ["Idx\tRatio\tTrain\tVal\tName"]
    lines += [f"{i}\t{1.0 / i:.4f}\t1\t1\tclass{i}, synonym" for i in range(1, 151)]
    (root / "objectInfo150.txt").write_text("\n".join(lines) + "\n")
    for folder in ("images/validation", "annotations/validation"):
        (root / folder).mkdir(parents=True)
    for stem, label in images.items():
        Image.fromarray(np.zeros(label.shape + (3,), np.uint8)).save(root / "images/validation" / f"{stem}.jpg")
        Image.fromarray(label.astype(np.uint8)).save(root / "annotations/validation" / f"{stem}.png")


def _write_cityscapes(root: Path, labels: dict[str, np.ndarray]) -> None:
    """Cityscapes layout with raw labelIds (7 = road, 11 = building, 26 = car, 0 = void)."""
    for stem, label in labels.items():
        city = stem.split("_")[0]
        for folder in ("leftImg8bit/val", "gtFine/val"):
            (root / folder / city).mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros(label.shape + (3,), np.uint8)).save(root / "leftImg8bit/val" / city / f"{stem}_leftImg8bit.png")
        Image.fromarray(label.astype(np.uint8)).save(root / "gtFine/val" / city / f"{stem}_gtFine_labelIds.png")


def _blocks(values: list[int], shape=(96, 128), block=32) -> np.ndarray:
    rows, cols = shape[0] // block, shape[1] // block
    grid = np.resize(np.asarray(values), rows * cols).reshape(rows, cols)
    return np.kron(grid, np.ones((block, block), dtype=int))


def _oracle_records(dataset, tmp_path: Path) -> list[dict]:
    """Paint each sample's ground truth with its prompted colours, as a perfect model would."""
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        truth = sample.ground_truth
        transform = CoordinateTransform.from_dict(sample.metadata["transform"])
        with Image.open(truth["label_path"]) as raw:
            gt = np.asarray(truth["label_lut"], np.uint8)[np.asarray(transform.transform_mask(raw))]
        painted = np.zeros(gt.shape + (3,), np.uint8)
        for class_id, colour in zip(truth["class_ids"], truth["colors"]):
            painted[gt == class_id] = colour
        path = tmp_path / f"{index:08d}_trial_000.png"
        Image.fromarray(painted).save(path)
        records.append({"sample": _Reference(sample), "output_paths": [path],
                        "trials": [{"status": "scored", "image": f"images/{path.name}"}]})
    return records


def test_named_palette_names_match_their_rgb_values():
    assert len(ade.NAMED_COLOURS) >= 27  # the most classes in one ADE20K val image
    assert len({rgb for _, rgb in ade.NAMED_COLOURS}) == len(ade.NAMED_COLOURS)
    # Everyday meanings where CSS differs: CSS green is (0, 128, 0), CSS darkgray is (169, 169, 169).
    intentional = {"green": (0, 255, 0), "dark gray": (64, 64, 64)}
    for name, rgb in ade.NAMED_COLOURS:
        assert intentional.get(name, ImageColor.getrgb(name.replace(" ", ""))) == rgb, name


def test_ade20k_oracle_scores_perfectly_and_collapses_to_groups(tmp_path):
    raw = tmp_path / "ade"
    _write_ade(raw, {"ADE_val_00000001": _blocks([1, 3, 4, 0]), "ADE_val_00000002": _blocks([2, 13, 0, 5])})
    dataset = ade.load_dataset({"raw_dir": str(raw), "mode": "single", "palette": "named", "size_multiple": 32})
    sample = dataset[0]
    assert sample.prompt.startswith("Convert this photo into a color-coded map: the class1 red")
    metrics = aggregator.aggregate(_oracle_records(dataset, tmp_path), {"refine": False, "bootstrap": 0})
    assert metrics["fine_miou"] == pytest.approx(1.0)
    assert metrics["fine_aacc"] == pytest.approx(1.0)
    assert metrics["coarse_fine_miou"] == pytest.approx(1.0)


@pytest.mark.parametrize("class_list", ["present", "all"])
def test_cityscapes_oracle_scores_perfectly(tmp_path, class_list):
    raw = tmp_path / "cityscapes"
    _write_cityscapes(raw, {"aachen_000000_000019": _blocks([7, 11, 26, 0]), "bremen_000001_000019": _blocks([11, 7])})
    dataset = cityscapes.load_dataset({"raw_dir": str(raw), "class_list": class_list, "size_multiple": 32})
    listed = dataset[0].ground_truth["class_ids"]
    assert listed == ([0, 2, 13] if class_list == "present" else list(range(19)))
    metrics = aggregator.aggregate(_oracle_records(dataset, tmp_path), {"refine": False, "bootstrap": 0})
    assert metrics["fine_miou"] == pytest.approx(1.0)


def test_frequency_order_ignores_this_images_areas(tmp_path):
    raw = tmp_path / "ade"
    # Same two classes, opposite areas; class 1 is the more frequent class dataset-wide.
    _write_ade(raw, {"ADE_val_00000001": _blocks([1, 1, 1, 2]), "ADE_val_00000002": _blocks([2, 2, 2, 1])})
    by_frequency = ade.load_dataset({"raw_dir": str(raw), "mode": "single", "palette": "named"})
    by_area = ade.load_dataset({"raw_dir": str(raw), "mode": "single", "palette": "named", "color_order": "area"})
    assert by_frequency[0].ground_truth["colors"] == by_frequency[1].ground_truth["colors"]
    assert by_area[0].ground_truth["colors"] != by_area[1].ground_truth["colors"]


def test_max_pixels_only_shrinks_oversized_images():
    assert ade._area_limit((683, 512), (1024, 1024), 524288) == (1024, 1024)
    limit = ade._area_limit((2000, 2000), (1024, 1024), 524288)
    assert limit[0] * limit[1] <= 524288


def test_subset_must_list_existing_images(tmp_path, monkeypatch):
    raw = tmp_path / "ade"
    _write_ade(raw, {"ADE_val_00000001": _blocks([1, 2])})
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    (subsets / "bad.txt").write_text("ADE_val_00000001\nADE_val_99999999\n")
    monkeypatch.setattr(ade, "read_subset", lambda name: {
        line for line in (subsets / name).read_text().splitlines() if line})
    with pytest.raises(ValueError, match="missing"):
        ade.load_dataset({"raw_dir": str(raw), "subset_file": "bad.txt"})


def test_decode_maps_black_to_ignore_and_colours_to_classes():
    image = Image.fromarray(np.array([[[250, 5, 5], [0, 0, 0]], [[3, 250, 0], [10, 10, 10]]], np.uint8))
    decoded = evaluator.decode(image, [4, 7], [(255, 0, 0), (0, 255, 0)])
    assert decoded.tolist() == [[4, evaluator.IGNORE], [7, evaluator.IGNORE]]


def test_refine_removes_small_blobs():
    pred = np.zeros((40, 40), np.int32)
    pred[10:13, 10:13] = 1  # 9-pixel speck inside class 0
    assert (evaluator.refine(pred, n_classes=2) == 0).all()
