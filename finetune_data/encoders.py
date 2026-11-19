"""Render canonical annotations as RGB target images (the y of each (x, t, y) triplet).

Defaults match how our evaluators decode model outputs:
- segmentation: tasks/evaluators/semseg.py (nearest prompted colour, black = no class)
- depth: tasks/evaluators/nyuv2_depth.py (BT.709 luminance, affine fit in disparity space)
- normals: tasks/evaluators/surface_normal.py with rgb_to_gt ["-r", "+g", "+b"]
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def segmentation(label: np.ndarray, palette: list[tuple[str, str, tuple[int, int, int]]]) -> Image.Image:
    """label holds class ids 1..len(palette) (0 or out of range = void, painted black)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (_, _, rgb) in enumerate(palette, start=1):
        lut[class_id] = rgb
    return Image.fromarray(lut[np.clip(label, 0, 255).astype(np.uint8)])


def depth(depth_m: np.ndarray, *, encoding: str = "disparity", percentiles=(1.0, 99.0)) -> Image.Image:
    """Grayscale, near = white and far = black, normalised per image.

    disparity (default): brightness linear in 1/depth, which the depth evaluator
      recovers exactly with its per-image affine fit in disparity space.
    linear: brightness linear in depth (the evaluator's *_depthspace metrics).
    Invalid pixels (depth <= 0) are black.
    """
    valid = depth_m > 0
    gray = np.zeros(depth_m.shape, dtype=np.float64)
    if valid.any():
        if encoding == "disparity":
            value = np.where(valid, 1.0 / np.where(valid, depth_m, 1.0), 0.0)
        elif encoding == "linear":
            value = np.where(valid, -depth_m, 0.0)
        else:
            raise ValueError("depth encoding must be 'disparity' or 'linear'")
        lo, hi = np.percentile(value[valid], percentiles)
        gray = np.where(valid, np.clip((value - lo) / max(hi - lo, 1e-12), 0.0, 1.0), 0.0)
    return Image.fromarray(np.round(gray * 255).astype(np.uint8)).convert("RGB")


def normals(xyz: np.ndarray, valid: np.ndarray | None = None) -> Image.Image:
    """Camera-space unit normals in the evaluators' ground-truth frame.

    R = (1 - x) / 2, G = (1 + y) / 2, B = (1 + z) / 2: the Vision Banana encoding, and
    the inverse of surface_normal.py's calibrated rgb_to_gt ["-r", "+g", "+b"].
    """
    rgb = np.stack([(1 - xyz[..., 0]) / 2, (1 + xyz[..., 1]) / 2, (1 + xyz[..., 2]) / 2], axis=-1)
    rgb = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0
    return Image.fromarray(rgb)
