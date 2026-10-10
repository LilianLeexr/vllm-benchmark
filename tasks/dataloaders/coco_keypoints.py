"""COCO 2017 keypoint loader for single-person pose-preservation experiments.

The loader deterministically selects the largest sufficiently visible annotated
person in every image, then carries its COCO keypoints through VisionBench's
image preprocessing transform.  This deliberately makes the first benchmark
single-person; it avoids ambiguous matching between generated and GT people.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from visionbench.transforms import CoordinateTransform, preprocess_image
from visionbench.types import Sample


COCO_PERSON_CATEGORY_ID = 1


@dataclass(frozen=True)
class _Record:
    image_path: Path
    image_id: int
    annotation: dict[str, Any]


class COCOSinglePersonKeypoints:
    def __init__(
        self,
        root: Path,
        *,
        annotations_file: str,
        images_dir: str,
        min_visible_keypoints: int,
        min_person_area: float,
        max_samples: int | None,
        max_size: tuple[int, int] | None,
        shrink_mode: str,
        size_multiple: int,
        prompt: str,
    ) -> None:
        annotation_path = root / annotations_file
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO keypoint annotations not found: {annotation_path}")
        image_root = root / images_dir
        if not image_root.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {image_root}")
        with annotation_path.open(encoding="utf-8") as stream:
            coco = json.load(stream)
        images = {int(image["id"]): image for image in coco["images"]}
        candidates: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            if (
                int(annotation.get("category_id", -1)) == COCO_PERSON_CATEGORY_ID
                and not annotation.get("iscrowd", 0)
                and int(annotation.get("num_keypoints", 0)) >= min_visible_keypoints
                and float(annotation.get("area", 0.0)) >= min_person_area
                and len(annotation.get("keypoints", [])) == 51
            ):
                candidates.setdefault(int(annotation["image_id"]), []).append(annotation)

        records: list[_Record] = []
        for image_id, people in candidates.items():
            image = images.get(image_id)
            if image is None:
                continue
            path = image_root / image["file_name"]
            if not path.is_file():
                raise FileNotFoundError(f"COCO image referenced by annotations is missing: {path}")
            # Stable tie-break: area, then annotation id.
            person = max(people, key=lambda item: (float(item["area"]), -int(item["id"])))
            records.append(_Record(path, image_id, person))
        self.records = sorted(records, key=lambda item: item.image_id)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise ValueError("No COCO people passed the configured visibility/area filters")
        self.max_size = max_size
        self.shrink_mode = shrink_mode
        self.size_multiple = size_multiple
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        processed, transform, preprocessing = preprocess_image(
            image, max_size=self.max_size, shrink_mode=self.shrink_mode, size_multiple=self.size_multiple
        )
        ground_truth = _ground_truth(record.annotation, transform)
        return Sample(
            sample_id=f"coco_val2017_{record.image_id:012d}",
            image=processed,
            prompt=self.prompt,
            ground_truth=ground_truth,
            metadata={
                "source_path": str(record.image_path),
                "coco_image_id": record.image_id,
                "coco_annotation_id": int(record.annotation["id"]),
                "transform": transform.to_dict(),
                "preprocessing": preprocessing,
            },
        )


def _ground_truth(annotation: dict[str, Any], transform: CoordinateTransform) -> dict[str, Any]:
    original = annotation["keypoints"]
    keypoints: list[list[float]] = []
    for offset in range(0, len(original), 3):
        x, y, visibility = float(original[offset]), float(original[offset + 1]), int(original[offset + 2])
        x, y = transform(x, y)
        # COCO v=1 means labelled but occluded; it is still a valid target.
        if visibility > 0 and not transform.visible((original[offset], original[offset + 1])):
            visibility = 0
        keypoints.append([x, y, visibility])
    bbox = transform.crop_box_to_visible((
        float(annotation["bbox"][0]), float(annotation["bbox"][1]),
        float(annotation["bbox"][0]) + float(annotation["bbox"][2]),
        float(annotation["bbox"][1]) + float(annotation["bbox"][3]),
    ))
    if bbox is None:
        raise ValueError(f"Preprocessing removed the selected person's bounding box (annotation {annotation['id']})")
    return {"keypoints": keypoints, "bbox_xyxy": list(bbox), "coco_area": float(annotation["area"])}


def load_dataset(config: dict) -> COCOSinglePersonKeypoints:
    max_size = config.get("max_size", (1024, 1024))
    limit = tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size
    max_samples = config.get("max_samples")
    if max_samples is not None and (not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer when supplied")
    return COCOSinglePersonKeypoints(
        Path(config["raw_dir"]),
        annotations_file=config.get("annotations_file", "annotations/person_keypoints_val2017.json"),
        images_dir=config.get("images_dir", "val2017"),
        min_visible_keypoints=int(config.get("min_visible_keypoints", 10)),
        min_person_area=float(config.get("min_person_area", 4096)),
        max_samples=max_samples,
        max_size=limit,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=int(config.get("size_multiple", 32)),
        prompt=str(config.get("prompt", "Keep the person and their pose unchanged. Return one image.")),
    )
