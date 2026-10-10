"""Pose-preservation metrics using a fixed pretrained COCO pose estimator."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from PIL import Image

from visionbench.types import Sample


METRICS = {
    "pck_at_005": {"unit": "fraction", "higher_is_better": True},
    "pck_at_010": {"unit": "fraction", "higher_is_better": True},
    "pck_at_020": {"unit": "fraction", "higher_is_better": True},
    "normalized_mean_keypoint_error": {"unit": "fraction", "higher_is_better": False},
    "pose_detection_failure_rate": {"unit": "fraction", "higher_is_better": False},
    "missing_keypoint_rate": {"unit": "fraction", "higher_is_better": False},
    "coco_oks": {"unit": "fraction", "higher_is_better": True},
}

# Official COCO keypoint sigmas, in the standard 17-keypoint order.
COCO_SIGMAS = (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
               0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089)


def prepare_output(image: Image.Image, sample: Sample, config: dict) -> Image.Image:
    """Align provider output with the processed input coordinate frame before scoring."""
    if image.size == sample.image.size:
        return image.convert("RGB")
    return image.convert("RGB").resize(sample.image.size, Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def _detector() -> Any:
    try:
        import torch
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
    except ImportError as exc:
        raise RuntimeError("coco_pose requires torch and torchvision") from exc
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).eval()
    return model, weights, torch


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = ((first[2] - first[0]) * (first[3] - first[1]) +
             (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def _predict(
    image: Image.Image, score_threshold: float, gt_bbox: list[float], match_iou_threshold: float
) -> tuple[list[list[float]], list[float]] | None:
    model, weights, torch = _detector()
    tensor = weights.transforms()(image)
    with torch.inference_mode():
        result = model([tensor])[0]
    boxes = result["boxes"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    keypoints = result["keypoints"].detach().cpu().tolist()
    keypoint_scores = result.get("keypoints_scores")
    keypoint_scores = keypoint_scores.detach().cpu().tolist() if keypoint_scores is not None else None
    candidates = [index for index, (label, score) in enumerate(zip(labels, scores)) if label == 1 and score >= score_threshold]
    if not candidates:
        return None
    # Match generated people to the single GT person in the shared image frame.
    best = max(candidates, key=lambda index: (_iou(boxes[index], gt_bbox), scores[index]))
    if _iou(boxes[best], gt_bbox) < match_iou_threshold:
        return None
    confidences = keypoint_scores[best] if keypoint_scores is not None else [1.0] * len(keypoints[best])
    return keypoints[best], confidences


def _scale(bbox: list[float], mode: str) -> float:
    width, height = max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])
    if mode == "bbox_diagonal":
        return max(1.0, math.hypot(width, height))
    if mode == "sqrt_bbox_area":
        return max(1.0, math.sqrt(width * height))
    raise ValueError("normalization must be 'bbox_diagonal' or 'sqrt_bbox_area'")


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    gt = sample.ground_truth
    if not isinstance(gt, dict) or len(gt.get("keypoints", [])) != 17:
        raise ValueError("coco_pose requires 17 COCO keypoints in sample.ground_truth")
    thresholds = tuple(float(value) for value in config.get("pck_thresholds", (0.05, 0.10, 0.20)))
    if thresholds != (0.05, 0.10, 0.20):
        raise ValueError("This evaluator declares PCK metrics only at 0.05, 0.10, and 0.20")
    prediction = _predict(
        generated_image,
        float(config.get("detection_score_threshold", 0.5)),
        list(gt["bbox_xyxy"]),
        float(config.get("match_iou_threshold", 0.1)),
    )
    visible = [index for index, point in enumerate(gt["keypoints"]) if int(point[2]) > 0]
    if not visible:
        raise ValueError("Selected GT person has no visible keypoints after preprocessing")
    if prediction is None:
        return {"pck_at_005": 0.0, "pck_at_010": 0.0, "pck_at_020": 0.0,
                "normalized_mean_keypoint_error": 1.0, "pose_detection_failure_rate": 1.0,
                "missing_keypoint_rate": 1.0, "coco_oks": 0.0}
    predicted, keypoint_confidences = prediction
    person_scale = _scale(gt["bbox_xyxy"], str(config.get("normalization", "bbox_diagonal")))
    confidence_threshold = float(config.get("keypoint_score_threshold", 0.2))
    distances = [math.dist(predicted[index][:2], gt["keypoints"][index][:2]) / person_scale for index in visible]
    valid = [distance for index, distance in zip(visible, distances)
             if math.isfinite(distance) and keypoint_confidences[index] >= confidence_threshold]
    missing = len(distances) - len(valid)
    if not valid:
        return {"pck_at_005": 0.0, "pck_at_010": 0.0, "pck_at_020": 0.0,
                "normalized_mean_keypoint_error": 1.0, "pose_detection_failure_rate": 0.0,
                "missing_keypoint_rate": 1.0, "coco_oks": 0.0}
    squared_area = max(1.0, (gt["bbox_xyxy"][2] - gt["bbox_xyxy"][0]) * (gt["bbox_xyxy"][3] - gt["bbox_xyxy"][1]))
    oks = sum(math.exp(-(distance * person_scale) ** 2 / (2 * squared_area * COCO_SIGMAS[index] ** 2))
              for index, distance in zip(visible, distances)
              if math.isfinite(distance) and keypoint_confidences[index] >= confidence_threshold) / len(visible)
    return {
        "pck_at_005": sum(distance < 0.05 for distance in valid) / len(visible),
        "pck_at_010": sum(distance < 0.10 for distance in valid) / len(visible),
        "pck_at_020": sum(distance < 0.20 for distance in valid) / len(visible),
        "normalized_mean_keypoint_error": sum(valid) / len(valid),
        "pose_detection_failure_rate": 0.0,
        "missing_keypoint_rate": missing / len(visible),
        "coco_oks": oks,
    }
