"""Affine-invariant zero-shot depth metrics for grayscale depth visualizations.

Follows RINO (arXiv:2607.12450), which reuses the Marigold / Depth Anything
protocol. The editor is asked for near = bright, far = dark, so BT.709 luminance
is a disparity-like relative map. Per image:

1. Map the generated image back to the 640x480 ground-truth frame.
2. Valid pixels: ``min_depth < gt < max_depth`` inside the Eigen crop.
3. Headline (RINO Table 1): least-squares ``s * luma + t ~= 1 / gt`` in disparity
   space, invert to depth, clip to ``[min_depth, max_depth]``.
4. ``*_depthspace`` (RINO Table 2 / Marigold): least squares ``s * luma + t ~= gt``
   in linear depth; a negative ``s`` recovers the inverted polarity.

Metrics are per-image means, averaged over images by the default aggregator.
"""

import numpy as np
from PIL import Image

from visionbench.transforms import CoordinateTransform
from visionbench.types import Sample

_LUMA = np.array([0.2126, 0.7152, 0.0722])
# Eigen et al. evaluation crop on the 480x640 NYUv2 frame (rows, cols).
EIGEN_CROP = (45, 471, 41, 601)

_ERRORS = {"abs_rel": "ratio", "sq_rel": "metres", "rmse": "metres", "rmse_log": "log metres"}
_ACCURACY = ("delta1", "delta2", "delta3")
METRICS = {
    **{name: {"unit": unit, "higher_is_better": False} for name, unit in _ERRORS.items()},
    **{name: {"unit": "fraction", "higher_is_better": True} for name in _ACCURACY},
    **{f"{name}_depthspace": {"unit": unit, "higher_is_better": False} for name, unit in _ERRORS.items()},
    **{f"{name}_depthspace": {"unit": "fraction", "higher_is_better": True} for name in _ACCURACY},
}


def decode_luminance(image: Image.Image) -> np.ndarray:
    """Generated RGB image -> (H, W) relative map in [0, 1]; brighter = nearer."""
    return (np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0) @ _LUMA


def to_ground_truth_frame(pred: np.ndarray, transform: CoordinateTransform) -> np.ndarray:
    """Undo the loader's scale/crop; pixels the model never saw become NaN."""
    output_w, output_h = transform.output_size
    if pred.shape != (output_h, output_w):  # models may return another size
        pred = np.asarray(Image.fromarray(pred.astype(np.float32)).resize(
            (output_w, output_h), Image.Resampling.BILINEAR))
    scaled_w, scaled_h = transform.scaled_size
    left, top, right, bottom = transform.final_crop
    scaled = np.full((scaled_h, scaled_w), np.nan, dtype=np.float32)
    scaled[top:bottom, left:right] = pred
    x0, y0, x1, y1 = transform.source_crop
    seen = np.asarray(Image.fromarray(np.isfinite(scaled).astype(np.uint8) * 255).resize(
        (x1 - x0, y1 - y0), Image.Resampling.NEAREST)) > 0
    filled = np.asarray(Image.fromarray(np.nan_to_num(scaled).astype(np.float32)).resize(
        (x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
    original_w, original_h = transform.original_size
    out = np.full((original_h, original_w), np.nan, dtype=np.float64)
    out[y0:y1, x0:x1] = np.where(seen, filled, np.nan)
    return out


def _fit(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.stack([pred, np.ones_like(pred)], axis=-1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(scale), float(shift)


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Standard metrics on already aligned, positive depth (1-D arrays of valid pixels)."""
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "sq_rel": float(np.mean((pred - gt) ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def score_relative(pred: np.ndarray, gt: np.ndarray, *, min_depth: float = 1e-3, max_depth: float = 10.0,
                   eigen_crop: bool = True, min_valid_pixels: int = 10) -> dict[str, float]:
    """Score a relative, disparity-like ``pred`` against metric ``gt`` of the same shape."""
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred)
    if eigen_crop:
        crop = np.zeros_like(valid)
        row0, row1, col0, col1 = EIGEN_CROP
        crop[row0:row1, col0:col1] = True
        valid &= crop
    if valid.sum() < min_valid_pixels:
        return {}
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)

    scale, shift = _fit(p, 1.0 / g)
    disparity = np.clip(scale * p + shift, 1e-6, None)
    scores = depth_metrics(np.clip(1.0 / disparity, min_depth, max_depth), g)

    scale, shift = _fit(p, g)
    linear = depth_metrics(np.clip(scale * p + shift, min_depth, max_depth), g)
    scores.update({f"{name}_depthspace": value for name, value in linear.items()})
    return scores


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    gt = np.asarray(sample.ground_truth["depth"], dtype=np.float64)
    transform = CoordinateTransform.from_dict(sample.metadata["transform"])
    pred = to_ground_truth_frame(decode_luminance(generated_image), transform)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction {pred.shape} does not match ground truth {gt.shape}")
    return score_relative(
        pred, gt,
        min_depth=float(config.get("min_depth", 1e-3)),
        max_depth=float(config.get("max_depth", 10.0)),
        eigen_crop=bool(config.get("eigen_crop", True)),
    )
