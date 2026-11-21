"""Instructions (the t of each triplet). Defaults are the exact text our evaluators send
to the model, including any system prompt the backend prepends, so training and testing match."""

from __future__ import annotations

SEGMENTATION_TEMPLATE = "Convert this photo into a color-coded map: {class_color_list}, and everything else black."

DEPTH = (  # tasks/dataloaders/nyuv2_depth.py DEFAULT_PROMPT; its config has an empty system prompt
    "Convert this image into a realistic grayscale depth visualization where each pixel's "
    "brightness indicates its distance from the camera. Ensure nearby foreground objects are "
    "bright, background areas are dark, and depth changes remain smooth."
)

# configs/qwen_normal_*.yaml system prompt + tasks/dataloaders/_surface_normal_common.py
# DEFAULT_PROMPT, joined the way visionbench/models/qwen.py joins them.
NORMALS = (
    "Edit the input image according to the instruction. Return one image.\n\n"
    "Convert this photo into a surface normal map of exactly the same scene. "
    "Keep the camera viewpoint, layout, object boundaries and image size unchanged. "
    "Color every pixel by the 3D direction its surface faces, in the standard RGB "
    "normal-map encoding (R = x, G = y, B = z, each mapped from [-1, 1] to [0, 255]). "
    "Output only the normal map: no text, no shading, no original photo colors."
)


def colour_text(name: str, rgb, colour_format: str) -> str:
    if colour_format == "name":
        return name
    if colour_format == "rgb":
        return f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"
    if colour_format == "hex":
        return "#{:02X}{:02X}{:02X}".format(*rgb)
    raise ValueError("colour_format must be 'name', 'rgb' or 'hex'")


def segmentation(palette, class_ids: list[int], *, colour_format: str = "name",
                 template: str = SEGMENTATION_TEMPLATE) -> str:
    """class_ids are 1-based ids into palette, listed in the given order."""
    parts = [f"the {palette[i - 1][0]} {colour_text(palette[i - 1][1], palette[i - 1][2], colour_format)}"
             for i in class_ids]
    return template.replace("{class_color_list}", ", ".join(parts))


# RINO's human-pose prompt (tasks/evaluators/_rino_pose.py), used for training and evaluation.
def _rino_pose_prompt() -> str:
    from pathlib import Path
    from visionbench.plugins import load_module
    return load_module(Path(__file__).resolve().parents[1] / "tasks" / "evaluators" / "_rino_pose.py").PROMPT


POSE = _rino_pose_prompt()
