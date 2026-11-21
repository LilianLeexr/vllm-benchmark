"""Human pose estimation from a generated OpenPose-style skeleton (RINO protocol).

Each output is resized to the 512x768 canvas (as RINO does), decoded to COCO-17 keypoints, mapped
back to crop pixels and compared with the GT. Two decoders are scored on the same image:

  rino_*    RINO's colour-free decoder (upright template matching) - comparable with RINO's numbers.
            Its ceiling is low: GT skeletons drawn perfectly score only ~0.37 PCK@0.1 on COCO val,
            because non-upright / side-view poses are mislabelled by the template.
  colour_*  ours: joints identified by their OpenPose colours (ceiling ~0.99 on perfect drawings),
            meaningful once a model is trained to reproduce the palette.

Per sample this returns pooled counts (n_vis, correct joints at each threshold, raw and PA-aligned)
and OKS; tasks/aggregators/pose_skeleton.py micro-averages them into PCK like RINO's metrics.average.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.types import Sample

_spec = importlib.util.spec_from_file_location("visionbench_task_rino_pose", Path(__file__).with_name("_rino_pose.py"))
rino = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rino)

DECODERS = {"rino": rino.decode_pose, "colour": rino.decode_pose_colour}
COUNTS = ("c05", "c10", "c20", "p05", "p10", "p20")
METRICS = {"n_vis": {"unit": "keypoints", "higher_is_better": True}}
for _d in DECODERS:
    METRICS.update({f"{_d}_{c}": {"unit": "keypoints", "higher_is_better": True} for c in COUNTS})
    METRICS[f"{_d}_oks"] = {"unit": "fraction", "higher_is_better": True}


def prepare_output(image: Image.Image, sample: Sample, config: dict) -> Image.Image:
    canvas = tuple(sample.metadata.get("canvas", rino.CANVAS))
    image = image.convert("RGB")
    return image if image.size == canvas else image.resize(canvas, Image.BILINEAR)


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    truth = sample.ground_truth
    gt = np.asarray(truth["keypoints"], dtype=np.float32)
    box = truth["letterbox"]
    out: dict[str, float] = {}
    for name, decode in DECODERS.items():
        pred = rino.unletterbox_kpts(decode(generated_image), box["scale"], box["ox"], box["oy"])
        row = rino.metrics(pred, gt, tuple(truth["bbox_wh"]))
        if not row:
            return {}
        out["n_vis"] = row["n_vis"]
        out.update({f"{name}_{c}": row[c] for c in COUNTS})
        out[f"{name}_oks"] = row["oks"]
    return out
