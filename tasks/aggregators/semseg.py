"""Dataset-level mIoU / mAcc / aAcc for colour-coded segmentation (RINO protocol).

IoU is computed once from pixel counts summed over the whole dataset, not averaged
per image. Generated images are re-decoded here, so this also implements RINO's
hierarchical fusion (coarse 10-group pass constrains the fine 150-class decode)
and its optional refinement.

Reported metrics (fractions; *_refined after mode filter + small-blob removal;
*_miou_ci95_low/high is a 95% bootstrap interval over images; <variant>/iou/<class>
gives per-class IoU for unrefined variants):
    fine_*          single fine pass
    hier_*          coarse + fine fused (hierarchical mode only)
    coarse_pass_*   the 10-group pass against GT mapped to 10 groups
    coarse_hier_*   fused prediction collapsed to 10 groups
    coarse_fine_*   single fine prediction collapsed to the coarse groups (no extra pass)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.plugins import load_module

ev = load_module(Path(__file__).resolve().parents[1] / "evaluators" / "semseg.py")
IGNORE = ev.IGNORE


class Totals:
    """Per-image pixel counts for one prediction variant."""

    def __init__(self, class_names: list[str]):
        self.class_names = list(class_names)
        self.rows: list[np.ndarray] = []  # per image: [inter (C), union (C), gt (C), correct, total]

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        c = ev.counts(pred, gt, len(self.class_names))
        self.rows.append(np.concatenate([c["intersection"], c["union"], c["gt"], [c["correct"], c["total"]]]))

    def _metrics(self, summed: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        n = len(self.class_names)
        inter, union, gt = summed[:n], summed[n:2 * n], summed[2 * n:3 * n]
        supported, labelled = union > 0, gt > 0
        iou = np.divide(inter, union, out=np.zeros(n), where=supported)
        miou = float(iou[supported].mean()) if supported.any() else 0.0
        macc = float((inter[labelled] / gt[labelled]).mean()) if labelled.any() else 0.0
        return miou, macc, float(summed[-2] / max(summed[-1], 1)), iou, supported

    def summary(self, prefix: str, *, bootstrap: int, per_class: bool) -> dict[str, float]:
        if not self.rows:
            return {}
        rows = np.stack(self.rows)
        miou, macc, aacc, iou, supported = self._metrics(rows.sum(axis=0))
        out = {f"{prefix}_miou": miou, f"{prefix}_macc": macc, f"{prefix}_aacc": aacc,
               f"{prefix}_images": float(len(rows))}
        if bootstrap and len(rows) > 1:
            # Resample images with replacement; 95% percentile interval of dataset-level mIoU.
            rng = np.random.default_rng(0)
            samples = [self._metrics(rows[rng.integers(0, len(rows), len(rows))].sum(axis=0))[0]
                       for _ in range(bootstrap)]
            out[f"{prefix}_miou_ci95_low"], out[f"{prefix}_miou_ci95_high"] = map(float, np.percentile(samples, [2.5, 97.5]))
        if per_class:
            out.update({f"{prefix}/iou/{name}": float(iou[c])
                        for c, name in enumerate(self.class_names) if supported[c]})
        return out


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
    bootstrap = config.get("bootstrap", 1000)
    do_refine = config.get("refine", True)
    window, min_area = config.get("refine_window", 5), config.get("refine_min_area", 100)
    by_image: dict[str, dict[str, dict]] = {}
    for record in records:
        if _scored_images(record):
            truth = record["sample"].ground_truth
            by_image.setdefault(truth["image_id"], {})[truth["stage"]] = record

    totals: dict[str, Totals] = {}

    def add(name: str, truth: dict, pred: np.ndarray, gt: np.ndarray, groups=None) -> None:
        """Count pred (and its refined version); optionally also collapsed to coarse groups."""
        variants = [("", pred)]
        if do_refine:
            variants.append(("_refined", ev.refine(pred, truth["n_classes"], window=window, min_area=min_area)))
        for suffix, value in variants:
            totals.setdefault(name + suffix, Totals(truth["class_names"])).add(value, gt)
            if groups is not None:
                lut, gt_groups, group_names = groups
                totals.setdefault(f"coarse_{name}{suffix}", Totals(group_names)).add(
                    np.where(value == IGNORE, IGNORE, lut[np.minimum(value, 255)]), gt_groups)

    for stages in by_image.values():
        fine, coarse = stages.get("fine"), stages.get("coarse")
        if fine is not None:
            sample = fine["sample"].load()
            truth = sample.ground_truth
            gt = ev.evaluable(ev.load_label(truth), sample)
            lut = np.asarray(truth["fine_to_coarse"] + [IGNORE] * (256 - len(truth["fine_to_coarse"])), dtype=np.int32)
            gt_groups = np.where(gt == IGNORE, IGNORE, lut[np.minimum(gt, 255)])
            # Also score the fine prediction collapsed to the coarse groups (coarse_fine_*).
            groups = (lut, gt_groups, truth["coarse_class_names"]) if "coarse_class_names" in truth else None
            for path in _scored_images(fine):
                pred = ev.decode(_open(path, sample), truth["class_ids"], truth["colors"], space=space)
                add("fine", truth, ev.to_original(pred, sample), gt, groups=groups)
        if coarse is not None:
            coarse_sample = coarse["sample"].load()
            coarse_truth = coarse_sample.ground_truth
            coarse_gt = ev.evaluable(ev.load_label(coarse_truth), coarse_sample)
            coarse_preds = []
            for path in _scored_images(coarse):
                # RINO decodes the coarse pass in RGB and does not refine it.
                pred = ev.decode(_open(path, coarse_sample), coarse_truth["class_ids"], coarse_truth["colors"])
                coarse_preds.append(pred)
                totals.setdefault("coarse_pass", Totals(coarse_truth["class_names"])).add(
                    ev.to_original(pred, coarse_sample), coarse_gt)
            if fine is not None:
                # Pair trials by index; with trials: 1 this is one pair per image.
                for coarse_pred, path in zip(coarse_preds, _scored_images(fine)):
                    fused = fuse(coarse_pred, _open(path, sample), truth["class_ids"], truth["colors"],
                                 truth["fine_to_coarse"])
                    add("hier", truth, ev.to_original(fused, sample), gt,
                        groups=(lut, gt_groups, coarse_truth["class_names"]))

    overall: dict[str, float] = {}
    for name, total in totals.items():
        overall.update(total.summary(name, bootstrap=bootstrap, per_class=not name.endswith("_refined")))
    return overall
