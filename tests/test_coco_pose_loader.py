import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


def _loader():
    path = Path(__file__).parents[1] / "tasks" / "dataloaders" / "coco_keypoints.py"
    spec = importlib.util.spec_from_file_location("coco_keypoints_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_coco_loader_selects_dominant_person_and_transforms_keypoints(tmp_path):
    root = tmp_path / "coco2017"
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.new("RGB", (100, 80), "white").save(root / "val2017" / "000000000001.jpg")
    keypoints = [20, 20, 2] * 17
    smaller = [10, 10, 2] * 17
    (root / "annotations" / "person_keypoints_val2017.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        "annotations": [
            {"id": 9, "image_id": 1, "category_id": 1, "iscrowd": 0, "num_keypoints": 17,
             "area": 2400, "bbox": [10, 10, 60, 40], "keypoints": keypoints},
            {"id": 3, "image_id": 1, "category_id": 1, "iscrowd": 0, "num_keypoints": 17,
             "area": 900, "bbox": [5, 5, 30, 30], "keypoints": smaller},
        ],
    }))
    dataset = _loader().load_dataset({
        "raw_dir": str(root), "min_visible_keypoints": 10, "min_person_area": 100,
        "max_size": [64, 64], "size_multiple": 32,
    })

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.sample_id == "coco_val2017_000000000001"
    assert sample.image.size == (64, 32)
    assert sample.ground_truth["bbox_xyxy"] == pytest.approx([6.4, 0.0, 44.8, 22.875])
    assert sample.ground_truth["keypoints"][0] == pytest.approx([12.8, 3.75, 2])
