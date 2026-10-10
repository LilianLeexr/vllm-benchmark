"""NYUv2 Eigen test split (654 images) for zero-shot monocular depth.

Expected layout under ``raw/nyuv2/`` (Marigold's ``nyu_labeled_extracted.tar``;
fetch it with ``scripts/download_nyuv2.sh``)::

    filename_list_test.txt            # "<rgb> <depth> <filled>" per line (optional)
    test/<scene>/rgb_XXXX.png         # 640x480 RGB
    test/<scene>/depth_XXXX.png       # uint16 raw depth in millimetres, 0 = invalid

Ground truth is the raw (not in-painted) sensor depth in metres, as in the
Marigold / Depth Anything evaluation protocol. The input image may be upscaled
so the editing model runs near its native resolution; the evaluator maps the
prediction back to the 640x480 ground-truth frame before scoring.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from visionbench.transforms import CoordinateTransform, preprocess_image
from visionbench.types import Sample

# RINO (arXiv:2607.12450) prompt for Qwen image editors on every depth dataset.
DEFAULT_PROMPT = (
    "Convert this image into a realistic grayscale depth visualization where each pixel's "
    "brightness indicates its distance from the camera. Ensure nearby foreground objects are "
    "bright, background areas are dark, and depth changes remain smooth."
)


def _upscale(image: Image.Image, long_side: int, size_multiple: int):
    """Aspect-preserving upscale to ``long_side``, then center-crop to the multiple."""
    width, height = image.size
    factor = long_side / max(width, height)
    scaled = (max(1, round(width * factor)), max(1, round(height * factor)))
    output = (scaled[0] // size_multiple * size_multiple, scaled[1] // size_multiple * size_multiple)
    if 0 in output:
        raise ValueError(f"Image {scaled} is smaller than size_multiple={size_multiple}")
    left, top = (scaled[0] - output[0]) // 2, (scaled[1] - output[1]) // 2
    transform = CoordinateTransform(
        original_size=(width, height), source_crop=(0, 0, width, height), scaled_size=scaled,
        output_size=output, scale=(scaled[0] / width, scaled[1] / height),
        final_crop=(left, top, left + output[0], top + output[1]),
    )
    params = {"upscale_long_side": long_side, "size_multiple": size_multiple, "original_size": (width, height),
              "scaled_size": scaled, "final_crop": transform.final_crop, "output_size": output}
    return transform.transform_image(image, resample=Image.Resampling.LANCZOS), transform, params


class NYUv2DepthDataset:
    def __init__(self, root: Path, *, split_file: str | None, prompt: str, upscale_long_side: int | None,
                 max_size, shrink_mode: str, size_multiple: int, depth_scale: float, limit: int | None):
        self.root = root
        self.prompt = prompt
        self.upscale_long_side = upscale_long_side
        self.max_size = max_size
        self.shrink_mode = shrink_mode
        self.size_multiple = size_multiple
        self.depth_scale = depth_scale
        split = root / split_file if split_file else None
        if split is not None and split.is_file():
            pairs = []
            for line in split.read_text().splitlines():
                if line.strip():
                    rgb, depth = line.split()[:2]
                    pairs.append((root / rgb, root / depth))
        else:
            pairs = [(rgb, rgb.with_name(rgb.name.replace("rgb_", "depth_")))
                     for rgb in sorted((root / "test").glob("*/rgb_*.png"))]
        if limit is not None:
            pairs = pairs[:limit]
        if not pairs:
            raise ValueError(f"No NYUv2 test images found under {root}; run scripts/download_nyuv2.sh")
        missing = [str(path) for pair in pairs for path in pair if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} NYUv2 files are missing, e.g. {missing[0]}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Sample:
        rgb_path, depth_path = self.pairs[index]
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        with Image.open(depth_path) as source:
            depth = np.asarray(source, dtype=np.float32) / self.depth_scale
        if depth.shape != (image.height, image.width):
            raise ValueError(f"Depth {depth.shape} does not match image {image.size} for {rgb_path}")
        if self.upscale_long_side:
            image, transform, parameters = _upscale(image, self.upscale_long_side, self.size_multiple)
        else:
            image, transform, parameters = preprocess_image(
                image, max_size=self.max_size, shrink_mode=self.shrink_mode, size_multiple=self.size_multiple,
            )
        relative = rgb_path.relative_to(self.root)
        return Sample(
            sample_id=f"{relative.parent.name}_{rgb_path.stem.removeprefix('rgb_')}",
            image=image,
            prompt=self.prompt,
            ground_truth={"depth": depth},
            metadata={"source_path": str(relative), "depth_path": str(depth_path.relative_to(self.root)),
                      "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> NYUv2DepthDataset:
    max_size = config.get("max_size", (1024, 1024))
    return NYUv2DepthDataset(
        Path(config["raw_dir"]),
        split_file=config.get("split_file", "filename_list_test.txt"),
        prompt=config.get("prompt", DEFAULT_PROMPT),
        upscale_long_side=config.get("upscale_long_side"),
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 16),
        depth_scale=float(config.get("depth_scale", 1000.0)),
        limit=config.get("limit"),
    )
