# Fine-tuning data

Builds the union multi-task fine-tuning set as (x, t, y) triplets (input image, task
instruction, target rendered as an RGB image), as in our proposal. The pipeline has two layers:

1. **Canonical store** (`finetune_data.sources.*`, built once): original-resolution images and
   annotations in one standard form per source, with train / val / test splits.
2. **Rendering** (`finetune_data.dataset.FinetuneDataset`): turns the store into triplets on the
   fly. Image size, prompts, target encodings, tasks and sampling weights are config values, so
   changing them needs no rebuild.

Defaults reproduce our evaluators' preprocessing, prompts and decoding, so a fine-tuned model
is scored by the existing benchmark configs exactly like the zero-shot baseline.
`tests/test_finetune_data.py` checks this by decoding generated targets with the evaluators.

## Build

One command builds everything from the public sources (about 12 GB, ~2 hours, mostly downloads) and
checks the result against `finetune_data/checksums/`, so every machine gets pixel-identical data
(verified across macOS arm64 and Linux x86_64):

```bash
bash scripts/build_finetune_data.sh            # -> data/finetune_canonical; rerun to resume
uv run python -m finetune_data.verify data/finetune_canonical   # check an existing store
```

Per source, if you only need some of it:

```bash
bash scripts/download_nyuv2_train.sh                                   # ~2.8 GB -> raw/nyuv2_train/
uv run python -m finetune_data.sources.nyuv2 raw/nyuv2_train data/finetune_canonical
bash scripts/download_vkitti2.sh                                       # ~17 GB downloaded, ~1 GB kept -> raw/vkitti2/
uv run python -m finetune_data.sources.vkitti2 raw/vkitti2 data/finetune_canonical
uv run python -m finetune_data.sources.hypersim data/finetune_canonical --train-frames 4000   # ~17 GB streamed, ~8.6 GB kept; resumable
uv run python -m finetune_data.sources.coco_pose raw/coco2017 data/finetune_canonical          # 4,100 person crops, ~0.7 GB
```

The test sets are downloaded separately by the evaluation scripts (`scripts/download_*.sh`).
Changing a source's parameters (frame counts, stride, test scene, seed) changes the data, so
`verify` will report differences; that is expected.

## Use

```python
from finetune_data.dataset import FinetuneDataset

ds = FinetuneDataset()                                   # finetune_data/configs/default.yaml
ds = FinetuneDataset({"tasks": {"depth": {"enabled": False}}})   # or any override / YAML path
item = ds[0]          # {"input": PIL, "instruction": str, "target": PIL, "task", "source", "id", "split"}
order = ds.sample_indices(10_000)   # weighted, shuffled; use as a torch sampler
```

To write files instead (PNG pairs + `manifest.jsonl` with a `sample_weight` column):

```bash
uv run python -m finetune_data.export data/finetune_export --splits train
```

## Options (`configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `splits` | `[train]` | `train`, `val` (held out from train for checkpoint selection), `test` |
| `sources.<name>.weight` | 1 | share within each task; each (task, source) group is sampled evenly regardless of size |
| `sources.<name>.tasks` | all it has | restrict a source to some tasks (Hypersim: `[normals]`) |
| `tasks.<task>.enabled / weight` | all on, 1 | which tasks to train and their sampling share |
| `size` (+ `tasks.<task>.size`) | per task, as in the eval configs | `max_size`, `max_area`, `upscale_long_side`, `size_multiple` |
| `tasks.segmentation.class_list` | `all` | `all` lists every class of the label space (Vision Banana; our baseline protocol), `present` only the classes in the image |
| `tasks.segmentation.colour_format` | `name` | `name`, `rgb` or `hex` in the prompt |
| `tasks.depth.encoding` | `disparity` | grayscale, near = white; `disparity` or `linear` |
| `tasks.*.instruction / template` | eval prompts | custom instruction text |

## Encodings

| Task | Target | Matches |
|---|---|---|
| Segmentation | flat colours from `palette.json`, black = void | `tasks/evaluators/semseg.py` |
| Depth | grayscale, brightness linear in 1/depth, normalised per image (1st–99th percentile) | `tasks/evaluators/nyuv2_depth.py` |
| Normals | R = (1 − x)/2, G = (1 + y)/2, B = (1 + z)/2 (Vision Banana) | `surface_normal.py` with `rgb_to_gt: [-r, +g, +b]` |

`palette.json` is one global class-to-colour table: the same class has the same colour in every
label space, a colour name always means one RGB value, and Cityscapes classes keep the exact
colours of the Cityscapes evaluator. Regenerate it with `python -m finetune_data.palette`.

## In-domain evaluation

The in-domain test splits are read straight from the canonical store, with the training
renderer's geometry, prompts and colours (`tasks/dataloaders/finetune_semseg.py`, `finetune_depth.py`):

| Config | Test set | Notes |
|---|---|---|
| `configs/indomain_nyuv2_semseg_{present,all}.yaml` | NYUv2 test, 654 images, 40 classes | `present` = RINO, `all` = Vision Banana / training default |
| `configs/indomain_vkitti2_semseg_{present,all}.yaml` | VKITTI2 Scene02, 282 images, 10 classes | |
| `configs/indomain_vkitti2_depth.yaml` | VKITTI2 Scene02 | `max_depth: 80`, no Eigen crop |
| `configs/qwen_nyuv2_depth.yaml` (existing) | NYUv2 test, 654 images | same images; raw sensor depth, Marigold protocol |
| `configs/diode_depth.yaml` | DIODE val, 771 images (out-of-domain depth) | Marigold packaging and range (0.6–350 m); `scripts/download_diode_depth.sh`. iBims-1 depth needs the password-protected TUM share, so it is not scripted |
| `configs/coco_pose_estimation.yaml` | COCO val2017, 3,061 persons | RINO protocol; reports `rino/*` (RINO's decoder, comparable with RINO) and `colour/*` (decodes by joint colour) |

```bash
uv run visionbench run --config configs/indomain_vkitti2_depth.yaml
```

Set a new `generation_version` for every model or checkpoint. Scores are computed at the
original resolution, so even a perfect prediction at the 1024-wide model resolution loses a
little at class / depth boundaries on VKITTI2 (oracle: mIoU 0.94, AbsRel 0.03); NYUv2 is not
resized for segmentation (oracle mIoU 1.0). Pose: RINO's decoder matches joints to an upright frontal template without using colour, so even perfectly drawn GT skeletons score only ~0.37 PCK@0.1 on COCO val (side views and bent poses get mislabelled); the colour decoder reaches ~0.99 on the same drawings. Report `rino/*` for comparison with RINO and `colour/*` for what a fine-tuned model actually draws. Out-of-domain sets keep their existing configs
(Cityscapes, ADE20K, iBims / DIODE).

## Sources

| Source | Tasks | Splits | Notes |
|---|---|---|---|
| `nyuv2` | segmentation (NYU-40), depth | 745 train / 50 val / 654 test | official labeled set; test = the Eigen split our evaluators use |
| `vkitti2` | segmentation (10 Cityscapes classes), depth | 2230 train / 50 val / 282 test | Camera_0, 6 weather variations, every 5th frame; test = all of Scene02 (shortest scene); depth in cm, sky invalid |
| `hypersim` | normals (default); also NYU-40 segmentation and depth | 4000 train / 100 val | Marigold's filtered frames, spread evenly over 347 scenes; Marigold's normal preprocessing, so normals are in the eval GT frame |
| `coco_pose` | human pose estimation (RINO protocol) | 4000 train / 100 val | train2017 only, one person per image; person crop + 17 keypoints, letterboxed to 512x768 and drawn as an OpenPose skeleton when rendered |

VKITTI2 classes are mapped to Cityscapes names and colours (Tree/Vegetation -> vegetation,
Van -> car, GuardRail/Misc -> void). Change the test scene or frame stride with
`--test-scene` / `--stride` (only frames ending in 0 or 5 are downloaded, so the stride must be a multiple of 5).
Val frames come from the training scenes: use them for checkpoint selection, and test / out-of-domain sets for generalisation.
When evaluating VKITTI2 depth use `max_depth: 80` (KITTI convention), not NYU's 10 m.

Hypersim is used only for normals by default (`sources.hypersim.tasks: [normals]`); add `segmentation` / `depth`
to that list to also train on its indoor NYU-40 labels and depth. It has no test split: normals are tested on
NYUv2 / iBims / DIODE (`configs/qwen_normal_*.yaml`). Frames are read straight from the remote scene zips with
HTTP range requests (`finetune_data/remote_zip.py`), so nothing besides the kept frames is downloaded.

Pose follows RINO's human pose estimation (`tasks/evaluators/_rino_pose.py`, ported from RINO under MIT):
each person's GT box is expanded by 25% and cropped, the crop is letterboxed into a 512x768 black canvas,
and the target is an OpenPose-coloured skeleton (joint radius 6, bone width 2: drawn thinner or thicker,
the decoders recover fewer joints). The test set is COCO val2017 (`configs/coco_pose_estimation.yaml`).
This is not the pose-preservation task in `configs/coco_pose_qwen.yaml`, which asks the editor to keep the
photo unchanged; that one stays useful to check that fine-tuning does not damage editing.
Cityscapes and ADE20K train splits are deliberately excluded so segmentation results stay
comparable to the zero-shot baseline and Vision Banana.
