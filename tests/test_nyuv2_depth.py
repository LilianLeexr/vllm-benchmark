import numpy as np
import pytest
from PIL import Image

from visionbench.config import load_config
from visionbench.plugins import load_dataset, load_evaluator, load_module
from visionbench.transforms import CoordinateTransform


def _write_nyu(root, count=2, size=(640, 480)):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    depth = 1.0 + 4.0 * xx / width + 2.0 * yy / height  # 1 m .. 7 m
    lines = []
    for index in range(1, count + 1):
        scene = root / "test" / "kitchen_0004"
        scene.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (index * 40, 90, 160)).save(scene / f"rgb_{index:04d}.png")
        Image.fromarray((depth * 1000).astype(np.uint16)).save(scene / f"depth_{index:04d}.png")
        lines.append(f"test/kitchen_0004/rgb_{index:04d}.png test/kitchen_0004/depth_{index:04d}.png "
                     f"test/kitchen_0004/filled_{index:04d}.png")
    (root / "filename_list_test.txt").write_text("\n".join(lines) + "\n")
    return depth


def _plugins():
    config = load_config("qwen_nyuv2_depth")
    return (config.scoped_path("dataset", "nyuv2_depth.py"), config.scoped_path("evaluator", "nyuv2_depth.py"))


def _disparity_image(depth: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Ideal model output: near = bright, linear in disparity, at the generated size."""
    disparity = 1.0 / depth
    gray = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    image = Image.fromarray((gray * 255).round().astype(np.uint8)).convert("RGB")
    return image.resize(size, Image.Resampling.BILINEAR)


def test_loader_upscales_and_keeps_metric_ground_truth(tmp_path):
    depth = _write_nyu(tmp_path)
    loader, _ = _plugins()
    dataset = load_dataset(loader, {"raw_dir": str(tmp_path), "upscale_long_side": 1024, "size_multiple": 32})
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.sample_id == "kitchen_0004_0001"
    assert sample.image.size == (1024, 768)
    assert "grayscale depth" in sample.prompt
    np.testing.assert_allclose(sample.ground_truth["depth"], depth, atol=1e-3)
    transform = CoordinateTransform.from_dict(sample.metadata["transform"])
    assert transform.original_size == (640, 480) and transform.output_size == (1024, 768)


def test_loader_without_split_file_globs_test_folder(tmp_path):
    _write_nyu(tmp_path, count=3)
    (tmp_path / "filename_list_test.txt").unlink()
    loader, _ = _plugins()
    dataset = load_dataset(loader, {"raw_dir": str(tmp_path), "size_multiple": 32})
    assert len(dataset) == 3
    assert dataset[0].image.size == (640, 480)


@pytest.mark.parametrize("upscale", [None, 1024, 1000])
def test_ideal_prediction_scores_near_perfect(tmp_path, upscale):
    _write_nyu(tmp_path, count=1)
    loader, evaluator_path = _plugins()
    params = {"raw_dir": str(tmp_path), "size_multiple": 32}
    if upscale:
        params["upscale_long_side"] = upscale
    sample = load_dataset(loader, params)[0]
    evaluator = load_evaluator(evaluator_path, {})
    output = _disparity_image(sample.ground_truth["depth"], sample.image.size)
    scores = evaluator.score(evaluator.prepare_output(output, sample), sample)
    assert set(scores) == set(evaluator.metrics)
    assert scores["abs_rel"] < 0.01 and scores["delta1"] > 0.999


def test_alignment_recovers_polarity_but_not_structure(tmp_path):
    _write_nyu(tmp_path, count=1)
    loader, evaluator_path = _plugins()
    sample = load_dataset(loader, {"raw_dir": str(tmp_path), "size_multiple": 32})[0]
    evaluator = load_evaluator(evaluator_path, {})
    ideal = _disparity_image(sample.ground_truth["depth"], sample.image.size)
    good = evaluator.score(ideal, sample)
    # A negative least-squares scale absorbs inverted polarity (as in RINO / MiDaS).
    inverted = evaluator.score(Image.eval(ideal, lambda v: 255 - v), sample)
    assert inverted["abs_rel"] == pytest.approx(good["abs_rel"], abs=1e-6)
    noise = np.random.default_rng(0).integers(0, 256, (*sample.image.size[::-1], 3), dtype=np.uint8)
    bad = evaluator.score(Image.fromarray(noise), sample)
    assert bad["abs_rel"] > 10 * good["abs_rel"]
    assert bad["delta1"] < 0.8


def test_metrics_match_reference_formulas():
    module = load_module(_plugins()[1])
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 9.0, size=(480, 640))
    pred = 1.0 / (gt * rng.uniform(0.9, 1.1, size=gt.shape))  # noisy disparity
    scores = module.score_relative(pred, gt)
    row0, row1, col0, col1 = module.EIGEN_CROP
    g, p = gt[row0:row1, col0:col1].ravel(), pred[row0:row1, col0:col1].ravel()
    scale, shift = np.polyfit(p, 1.0 / g, 1)
    depth = np.clip(1.0 / np.clip(scale * p + shift, 1e-6, None), 1e-3, 10.0)
    assert scores["abs_rel"] == pytest.approx(np.mean(np.abs(depth - g) / g), rel=1e-6)
    assert scores["delta1"] == pytest.approx(np.mean(np.maximum(depth / g, g / depth) < 1.25), rel=1e-6)


def test_invalid_pixels_and_too_few_pixels():
    module = load_module(_plugins()[1])
    gt = np.zeros((480, 640))
    assert module.score_relative(np.ones_like(gt), gt) == {}
    gt[100:200, 100:200] = 3.0
    gt[150, 150] = 50.0  # beyond max_depth: ignored
    pred = np.where(gt > 0, 1.0 / np.maximum(gt, 1e-3), 0.0)
    pred[150, 150] = np.nan
    scores = module.score_relative(pred + np.random.default_rng(1).normal(0, 1e-3, gt.shape), gt)
    assert np.isfinite(scores["abs_rel"])
