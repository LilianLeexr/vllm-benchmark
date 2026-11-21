"""COCO val2017 human pose estimation, RINO protocol (top-down, one sample per person).

Every non-crowd person with >= 8 visible keypoints and area >= 64^2 (3,061 in val2017) is cropped by
its GT box expanded by 25% per side and letterboxed into a 512x768 black canvas; the editor is asked
for an OpenPose-style skeleton (RINO's prompt). Decoding and scoring: tasks/evaluators/pose_skeleton.py.
The crop / canvas / prompt are shared with the fine-tuning targets (tasks/evaluators/_rino_pose.py).

Not to be confused with coco_keypoints.py (pose *preservation* during editing).

Expected layout under raw/coco2017/ (scripts/download_coco_pose.sh):
    val2017/*.jpg
    annotations/person_keypoints_val2017.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image

from visionbench.types import Sample

_spec = importlib.util.spec_from_file_location(
    "visionbench_task_rino_pose", Path(__file__).resolve().parents[1] / "evaluators" / "_rino_pose.py")
rino = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rino)


class COCOPoseEstimation:
    def __init__(self, root: Path, *, annotations_file: str, images_dir: str, prompt: str,
                 canvas: tuple[int, int], sample_stride: int, max_samples: int | None):
        annotation_path, image_root = root / annotations_file, root / images_dir
        if not annotation_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(f"expected {annotation_path} and {image_root}; run scripts/download_coco_pose.sh")
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {im["id"]: im["file_name"] for im in coco["images"]}
        samples = [{"path": image_root / names[a["image_id"]], "id": f"{a['image_id']}_{a['id']}",
                    "bbox": a["bbox"], "keypoints": a["keypoints"]}
                   for a in coco["annotations"] if rino.keep_annotation(a)]
        samples.sort(key=lambda s: s["id"])  # RINO's deterministic order
        samples = samples[::sample_stride]
        self.samples = samples[:max_samples] if max_samples else samples
        if not self.samples:
            raise ValueError("no COCO persons pass RINO's filter")
        self.prompt, self.canvas = prompt, canvas

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        s = self.samples[index]
        with Image.open(s["path"]) as image:
            crop, kpts, bbox_wh, box = rino.crop_person(image.convert("RGB"), s["bbox"], s["keypoints"])
        canvas, scale, ox, oy = rino.letterbox(crop, self.canvas)
        return Sample(
            sample_id=f"coco_val2017_{s['id']}",
            image=canvas,
            prompt=self.prompt,
            ground_truth={"keypoints": kpts.tolist(), "bbox_wh": list(bbox_wh),
                          "letterbox": {"scale": scale, "ox": ox, "oy": oy}},
            metadata={"source_path": str(s["path"]), "crop_box": list(box), "canvas": list(self.canvas)},
        )


def load_dataset(config: dict) -> COCOPoseEstimation:
    return COCOPoseEstimation(
        Path(config["raw_dir"]),
        annotations_file=config.get("annotations_file", "annotations/person_keypoints_val2017.json"),
        images_dir=config.get("images_dir", "val2017"),
        prompt=config.get("prompt") or rino.PROMPT,
        canvas=tuple(config.get("canvas", rino.CANVAS)),
        sample_stride=int(config.get("sample_stride", 1)),
        max_samples=config.get("max_samples"),
    )
