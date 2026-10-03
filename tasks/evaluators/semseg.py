"""Colour-coded semantic segmentation, following RINO (arXiv:2607.12450).

The loader asks the model to paint each class a flat colour. Every output pixel is
assigned to the nearest prompted colour (black = everything else), and the label
map is compared with the ground truth at its original resolution.

The loader stores everything needed for decoding in ``sample.ground_truth``, so this
evaluator is dataset-agnostic:
    label_path  original annotation PNG
    label_lut   256-entry list mapping raw PNG values to eval ids (255 = ignore)
    class_ids   eval ids listed in the prompt, in prompt order
    colors      RGB for each entry of class_ids
    n_classes   size of the eval label space
Per-image scores here are diagnostics only. Dataset-level mIoU, mAcc and aAcc are
computed from summed pixel counts in tasks/aggregators/semseg.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from visionbench.transforms import CoordinateTransform
from visionbench.types import Sample

IGNORE = 255

METRICS = {
    "pixel_acc": {"unit": "fraction", "higher_is_better": True},
    "image_miou": {"unit": "fraction", "higher_is_better": True},
}

_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) to CIELAB, shape (..., 3)."""
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _features(rgb: np.ndarray, space: str) -> np.ndarray:
    if space == "lab":
        return rgb_to_lab(rgb)
    if space == "rgb":
        return np.asarray(rgb, dtype=np.float32)
    raise ValueError("decode_space must be 'rgb' or 'lab'")


def nearest_slot(rgb: np.ndarray, colors: list, *, space: str = "rgb", black: bool = True) -> np.ndarray:
    """Index of the nearest colour per pixel of an (..., 3) array; len(colors) is the black sink."""
    rgb = np.asarray(rgb, dtype=np.float32)
    palette = np.asarray(list(colors) + ([(0, 0, 0)] if black else []), dtype=np.float32)
    pixels, centroids = _features(rgb.reshape(-1, 3), space), _features(palette, space)
    best = np.zeros(len(pixels), dtype=np.int32)
    best_distance = np.full(len(pixels), np.inf)
    for slot, centroid in enumerate(centroids):  # avoids an (N, K, 3) tensor
        distance = ((pixels - centroid) ** 2).sum(axis=1)
        closer = distance < best_distance
        best[closer], best_distance[closer] = slot, distance[closer]
    return best.reshape(rgb.shape[:-1])


def decode(image: Image.Image, class_ids: list[int], colors: list, *, space: str = "rgb") -> np.ndarray:
    """Nearest-colour decode to an eval-id map, IGNORE where black wins."""
    lookup = np.array(list(class_ids) + [IGNORE], dtype=np.int32)
    return lookup[nearest_slot(np.asarray(image.convert("RGB")), colors, space=space)]


def mode_filter(pred: np.ndarray, n_classes: int, window: int = 5) -> np.ndarray:
    """Local majority vote, matching RINO refine.mode_filter.

    Only classes present in ``pred`` can win; a window with no votes falls back to
    class 0, as RINO's argmax over all-zero votes does.
    """
    classes = [c for c in np.unique(pred) if c < n_classes]
    if not classes:
        return np.zeros_like(pred)
    votes = np.stack([ndi.uniform_filter((pred == c).astype(np.float32), size=window) for c in classes], axis=-1)
    out = np.asarray(classes, dtype=np.int32)[votes.argmax(axis=-1)]
    out[votes.max(axis=-1) <= 0] = 0
    return out


def remove_small_blobs(pred: np.ndarray, n_classes: int, min_area: int = 100) -> np.ndarray:
    """Drop connected components below min_area and refill them from a 3x3 mode."""
    out = pred.copy()
    holes = np.zeros(pred.shape, dtype=bool)
    for c in np.unique(pred):
        if c >= n_classes:
            continue
        labeled, count = ndi.label(pred == c, structure=np.ones((3, 3)))
        if count == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        small = np.flatnonzero(sizes < min_area)
        small = small[small > 0]
        if small.size:
            mask = np.isin(labeled, small)
            holes |= mask
            out[mask] = IGNORE
    if holes.any():
        filled = mode_filter(np.where(holes, 0, out), n_classes, window=3)
        out[holes] = filled[holes]
    return out


def refine(pred: np.ndarray, n_classes: int, *, window: int = 5, min_area: int = 100) -> np.ndarray:
    """RINO post-processing: mode filter, then small-blob removal."""
    return remove_small_blobs(mode_filter(pred, n_classes, window), n_classes, min_area)


def load_label(ground_truth: dict) -> np.ndarray:
    """Original-resolution eval-id map (IGNORE for void pixels)."""
    lut = np.asarray(ground_truth["label_lut"], dtype=np.uint8)
    with Image.open(ground_truth["label_path"]) as label:
        return lut[np.asarray(label, dtype=np.uint8)]


def to_original(pred: np.ndarray, sample: Sample) -> np.ndarray:
    """Map a processed-frame label map back onto the original image grid.

    Nearest-neighbour; pixels removed by the size-multiple crop become IGNORE so they
    are excluded from scoring.
    """
    transform = CoordinateTransform.from_dict(sample.metadata["transform"])
    width, height = transform.original_size
    xs, _ = np.asarray(transform.forward_points([(x + 0.5, 0) for x in range(width)])).T
    _, ys = np.asarray(transform.forward_points([(0, y + 0.5) for y in range(height)])).T
    xs, ys = np.floor(xs).astype(int), np.floor(ys).astype(int)
    out_width, out_height = transform.output_size
    valid_x, valid_y = (xs >= 0) & (xs < out_width), (ys >= 0) & (ys < out_height)
    out = pred[np.clip(ys, 0, out_height - 1)[:, None], np.clip(xs, 0, out_width - 1)[None, :]]
    out[~(valid_y[:, None] & valid_x[None, :])] = IGNORE
    return out


def counts(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict[str, np.ndarray | int]:
    """Per-class intersection / union / GT pixels over non-void GT pixels.

    A pixel predicted as black (IGNORE) counts as wrong, as in RINO metrics.counts.
    """
    valid = gt != IGNORE
    p, g = pred[valid].astype(np.int64), gt[valid].astype(np.int64)
    hit = p == g
    gt_count = np.bincount(g, minlength=n_classes)[:n_classes]
    inter = np.bincount(g[hit], minlength=n_classes)[:n_classes]
    pred_count = np.bincount(p[p < n_classes], minlength=n_classes)[:n_classes]
    return {"intersection": inter, "union": gt_count + pred_count - inter, "gt": gt_count,
            "correct": int(hit.sum()), "total": int(g.size)}


def evaluable(gt: np.ndarray, sample: Sample) -> np.ndarray:
    """GT with pixels outside the processed crop marked IGNORE."""
    outside = to_original(np.zeros(sample.image.size[::-1], dtype=np.uint8), sample) == IGNORE
    return np.where(outside, IGNORE, gt)


def prepare_output(image: Image.Image, sample: Sample, config: dict) -> Image.Image:
    # Bilinear would blend neighbouring class colours into a third colour.
    if image.size != sample.image.size:
        image = image.resize(sample.image.size, Image.Resampling.NEAREST)
    return image


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    truth = sample.ground_truth
    n_classes = truth["n_classes"]
    pred = decode(generated_image, truth["class_ids"], truth["colors"], space=config.get("decode_space", "rgb"))
    gt = evaluable(load_label(truth), sample)
    c = counts(to_original(pred, sample), gt, n_classes)
    supported = c["union"] > 0
    return {"pixel_acc": c["correct"] / max(c["total"], 1),
            "image_miou": float((c["intersection"][supported] / c["union"][supported]).mean()) if supported.any() else 0.0}
