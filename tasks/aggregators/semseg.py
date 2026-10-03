"""Dataset-level mIoU / mAcc / aAcc for colour-coded segmentation (RINO protocol).

IoU is computed once from pixel counts summed over the whole dataset, not averaged
per image. Generated images are re-decoded here, so this also implements RINO's
hierarchical fusion (coarse 10-group pass constrains the fine 150-class decode)
and its optional refinement.

Reported metrics (fractions; *_refined after mode filter + small-blob removal):
    fine_*          single fine pass
    hier_*          coarse + fine fused (hierarchical mode only)
    coarse_pass_*   the 10-group pass against GT mapped to 10 groups
    coarse_hier_*   fused prediction collapsed to 10 groups
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.plugins import load_module

ev = load_module(Path(__file__).resolve().parents[1] / "evaluators" / "semseg.py")
IGNORE = ev.IGNORE


class Totals:
    def __init__(self, n_classes: int):
        self.intersection = np.zeros(n_classes, dtype=np.int64)
        self.union = np.zeros(n_classes, dtype=np.int64)
        self.gt = np.zeros(n_classes, dtype=np.int64)
        self.correct = self.total = self.images = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        c = ev.counts(pred, gt, len(self.union))
        self.intersection += c["intersection"]
        self.union += c["union"]
        self.gt += c["gt"]
        self.correct += c["correct"]
        self.total += c["total"]
        self.images += 1

    def summary(self, prefix: str) -> dict[str, float]:
        if not self.images:
            return {}
        supported, labelled = self.union > 0, self.gt > 0
        return {
            f"{prefix}_miou": float((self.intersection[supported] / self.union[supported]).mean()),
            f"{prefix}_macc": float((self.intersection[labelled] / self.gt[labelled]).mean()),
            f"{prefix}_aacc": self.correct / max(self.total, 1),
        }


def _scored_images(record: dict) -> list[Path]:
    """Output images of trials that scored, so failed trials stay excluded."""
    by_name = {Path(path).name: Path(path) for path in record["output_paths"]}
    return [by_name[Path(row["image"]).name] for row in record["trials"]
            if row["status"] == "scored" and Path(row["image"]).name in by_name]


def _open(path: Path, sample) -> Image.Image:
    with Image.open(path) as image:
        return ev.prepare_output(image.copy(), sample, {})


def fuse(coarse: np.ndarray, fine_image: Image.Image, class_ids: list[int], colors: list,
         fine_to_coarse: list[int]) -> np.ndarray:
    """RINO hierarchical.fuse: each pixel takes the nearest fine colour (RGB, no black)
    among the classes of its coarse group; pixels the coarse pass left black stay IGNORE."""
    rgb = np.asarray(fine_image.convert("RGB"))
    out = np.full(coarse.shape, IGNORE, dtype=np.int32)
    slots_by_group: dict[int, list[int]] = {}
    for slot, class_id in enumerate(class_ids):
        slots_by_group.setdefault(fine_to_coarse[class_id], []).append(slot)
    for group, slots in slots_by_group.items():
        region = coarse == group
        if region.any():
            best = ev.nearest_slot(rgb[region], [colors[s] for s in slots], black=False)
            out[region] = np.asarray([class_ids[s] for s in slots], dtype=np.int32)[best]
    return out


def aggregate(records: list[dict], config: dict) -> dict[str, float]:
    space = config.get("decode_space", "rgb")
    do_refine = config.get("refine", True)
    window, min_area = config.get("refine_window", 5), config.get("refine_min_area", 100)
    by_image: dict[str, dict[str, dict]] = {}
    for record in records:
        if _scored_images(record):
            truth = record["sample"].ground_truth
            by_image.setdefault(truth["image_id"], {})[truth["stage"]] = record

    totals: dict[str, Totals] = {}

    def add(name: str, n_classes: int, pred: np.ndarray, gt: np.ndarray, groups=None) -> None:
        """Count pred (and its refined version); optionally also collapsed to coarse groups."""
        variants = [("", pred)]
        if do_refine:
            variants.append(("_refined", ev.refine(pred, n_classes, window=window, min_area=min_area)))
        for suffix, value in variants:
            totals.setdefault(name + suffix, Totals(n_classes)).add(value, gt)
            if groups is not None:
                lut, gt_groups, n_groups = groups
                totals.setdefault(f"coarse_{name}{suffix}", Totals(n_groups)).add(
                    np.where(value == IGNORE, IGNORE, lut[np.minimum(value, 255)]), gt_groups)

    for stages in by_image.values():
        fine, coarse = stages.get("fine"), stages.get("coarse")
        if fine is not None:
            sample = fine["sample"].load()
            truth = sample.ground_truth
            n_classes, gt = truth["n_classes"], ev.evaluable(ev.load_label(truth), sample)
            for path in _scored_images(fine):
                pred = ev.decode(_open(path, sample), truth["class_ids"], truth["colors"], space=space)
                add("fine", n_classes, ev.to_original(pred, sample), gt)
        if coarse is not None:
            coarse_sample = coarse["sample"].load()
            coarse_truth = coarse_sample.ground_truth
            coarse_gt = ev.evaluable(ev.load_label(coarse_truth), coarse_sample)
            coarse_preds = []
            for path in _scored_images(coarse):
                # RINO decodes the coarse pass in RGB and does not refine it.
                pred = ev.decode(_open(path, coarse_sample), coarse_truth["class_ids"], coarse_truth["colors"])
                coarse_preds.append(pred)
                totals.setdefault("coarse_pass", Totals(coarse_truth["n_classes"])).add(
                    ev.to_original(pred, coarse_sample), coarse_gt)
            if fine is not None:
                lut = np.asarray(truth["fine_to_coarse"] + [IGNORE] * (256 - len(truth["fine_to_coarse"])),
                                 dtype=np.int32)
                gt_groups = np.where(gt == IGNORE, IGNORE, lut[np.minimum(gt, 255)])
                # Pair trials by index; with trials: 1 this is one pair per image.
                for coarse_pred, path in zip(coarse_preds, _scored_images(fine)):
                    fused = fuse(coarse_pred, _open(path, sample), truth["class_ids"], truth["colors"],
                                 truth["fine_to_coarse"])
                    add("hier", n_classes, ev.to_original(fused, sample), gt,
                        groups=(lut, gt_groups, coarse_truth["n_classes"]))

    overall: dict[str, float] = {}
    for name, total in totals.items():
        overall.update(total.summary(name))
        if total.images:
            overall[f"{name}_images"] = float(total.images)
    return overall
