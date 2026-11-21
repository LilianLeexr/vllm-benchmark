# Fine-tuning data: union multi-task (x, t, y)

Training data for instruction-tuning Qwen-Image 2.1 on four vision tasks, in the format of our proposal:
an input image **x**, a text instruction **t**, and a target **y** rendered as an RGB image. Every target
is drawn exactly the way our evaluators decode model outputs, and every instruction is the exact prompt
the evaluation configs send, so a fine-tuned model is scored by the same configs as the zero-shot baseline.

| Task | Training source | Train / val | Test (in-domain) | Test (out-of-domain) |
|---|---|---|---|---|
| Semantic segmentation | NYUv2 (40 classes), Virtual KITTI 2 (10 Cityscapes classes) | 745 + 2230 / 50 + 50 | NYUv2 test (654), VKITTI2 Scene02 (282) | Cityscapes val, ADE20K val |
| Depth | NYUv2, Virtual KITTI 2 | 745 + 2230 / 50 + 50 | NYUv2 test (654), VKITTI2 Scene02 (282) | DIODE val (771) |
| Surface normals | Hypersim | 3965 / 100 | – (no normals training data from the test domains) | NYUv2, iBims-1, DIODE-indoor |
| Human pose estimation (RINO) | COCO train2017 | 4000 / 100 | COCO val2017 (3,061 persons) | – |

Contents: [1 Quick start](#1-quick-start) · [2 What a sample looks like](#2-what-a-sample-looks-like) ·
[3 Training with it](#3-training-with-it) · [4 Everything you can adjust](#4-everything-you-can-adjust) ·
[5 Targets and prompts](#5-targets-and-prompts) · [6 Evaluation](#6-evaluation) ·
[7 Decisions and known issues](#7-decisions-and-known-issues) · [8 Reproducibility and file layout](#8-reproducibility-and-file-layout) ·
[9 Code map](#9-code-map)

---

## 1. Quick start

```bash
# 1. Build all training data from the public sources (~12 GB, ~2 h, mostly downloads; rerun to resume).
#    It ends by checking the result against finetune_data/checksums/ (pixel-identical on every machine).
bash scripts/build_finetune_data.sh                    # -> data/finetune_canonical
#    or put it elsewhere:  ROOT=/ebs/finetune_canonical bash scripts/build_finetune_data.sh

# 2. Check an existing copy at any time
uv run python -m finetune_data.verify data/finetune_canonical
```

```python
from finetune_data.dataset import FinetuneDataset

ds = FinetuneDataset()                      # finetune_data/configs/default.yaml: all 4 tasks, train split
item = ds[0]                                # one (x, t, y) triplet, see section 2
order = ds.sample_indices(20_000, seed=0)   # balanced, shuffled indices (with replacement) for one run
```

Requirements: the repo's `uv` environment (adds `h5py`); about 12 GB for the store plus up to ~8 GB of
temporary downloads (VKITTI2 archives are deleted after extraction).

## 2. What a sample looks like

`ds[i]` returns a dict:

| Key | Type | Content |
|---|---|---|
| `input` | PIL RGB image | the photo, already resized for the model (sizes below) |
| `instruction` | str | the task prompt (identical to the evaluation prompt) |
| `target` | PIL RGB image | the answer drawn as an image, same size as `input` |
| `task` | str | `segmentation`, `depth`, `normals`, `pose` |
| `source` | str | `nyuv2`, `vkitti2`, `hypersim`, `coco_pose` |
| `split`, `id` | str | split name and a unique sample id |

Image sizes differ by task and source (all sides are multiples of 32, as Qwen-Image requires):

| Task / source | `input` = `target` size |
|---|---|
| segmentation, NYUv2 | 640 × 480 |
| segmentation, VKITTI2 | 1024 × 288 |
| depth, NYUv2 | 1024 × 768 (upscaled, as the NYUv2 depth eval) |
| depth, VKITTI2 | 1024 × 288 |
| normals, Hypersim | 1024 × 768 |
| pose, COCO | 512 × 768 (person crop letterboxed into a fixed canvas) |

So batches must group samples of equal size (bucket by `(task, source)` or by size), or use batch size 1
with gradient accumulation.

Default sampling (`sample_indices`): each task gets 25% of the samples, and within a task each source gets
an equal share regardless of how many images it has (e.g. NYUv2's 745 segmentation images are drawn as
often as VKITTI2's 2230). Change this with the weights in section 4.

## 3. Training with it

### 3.1 In Python (recommended)

```python
from finetune_data.dataset import FinetuneDataset, subset_config

train = FinetuneDataset()                                           # all tasks, train split
val = FinetuneDataset({"splits": ["val"]})                          # 300 held-out images, for checkpoint selection
depth_nyu = FinetuneDataset(subset_config(tasks=["depth"], sources=["nyuv2"]))   # one task / one source
custom = FinetuneDataset("my_config.yaml")                          # any keys from section 4, merged over the defaults
```

`FinetuneDataset` has `__len__` / `__getitem__`, so it plugs into a `torch.utils.data.DataLoader`; pass the
indices from `sample_indices(n)` as the sampler to get the balanced mix. Rendering happens on the fly (a few
ms per sample), so changing any option in section 4 needs no rebuild.

### 3.2 As files

```bash
uv run python -m finetune_data.export out/mixed --splits train              # everything
uv run python -m finetune_data.export out/depth_nyu --tasks depth --sources nyuv2
uv run python -m finetune_data.export out/per_lora --per-group               # one folder per (task, source)
uv run python -m finetune_data.export out/mixed --config my_config.yaml      # with your own options
```

Each folder gets `inputs/<task>/<id>.png`, `targets/<task>/<id>.png` and `manifest.jsonl` with
`id, source, task, split, instruction, input_path, target_path, sample_weight` (`sample_weight` reproduces the
balanced sampling; it sums to 1 over the file).

### 3.3 One shared adapter or one adapter per task?

Both are supported from the same data. Our recommendation:

- **Main experiment: one shared LoRA on the mixed data.** It is what the project claims (one model, the task
  chosen by the instruction), it is the setting Vision Banana reports, tasks can help each other (depth,
  normals and segmentation all need scene geometry), and there is only one adapter to check for lost editing
  ability.
- **Ablation: one LoRA per task** (`--per-group`, or `subset_config(tasks=[...])`). It tells us whether mixing
  helps or interferes, and a single small task (e.g. NYUv2 depth) is the quickest way to debug the training
  loop before the full mix.
- With the shared LoRA, log the validation loss per task; if one task dominates, adjust `tasks.<task>.weight`.

### 3.4 Things to watch while training

- **Do not add generic augmentations to pose or normals targets.** A horizontal flip must also flip the
  normal x component (in the target image: `R -> 255 - R`) and swap left/right joints in pose (the joint colours
  encode left/right). Flips are safe for segmentation and depth. Colour jitter must only touch `input`, never
  `target`. Resizing targets must use nearest neighbour (except depth). The safest choice is no augmentation;
  the data is already varied.
- **Normals' instruction already contains the system prompt** (`"Edit the input image according to the
  instruction. Return one image.\n\n..."`), joined exactly as the Qwen backend joins them at evaluation. Do not
  add a system prompt again for normals. The other tasks are evaluated with an empty system prompt.
- **Use `val`, not `test`, for checkpoint selection.** VKITTI2 / Hypersim / COCO val frames come from the
  training scenes or images, so they track fitting; generalisation is measured on the test and
  out-of-domain sets in section 6.
- **Watch editing ability.** `configs/coco_pose_qwen.yaml` (pose *preservation*, see 7.9) checks that the
  fine-tuned model can still edit photos without breaking them.

## 4. Everything you can adjust

### 4.1 Rendering options (`finetune_data/configs/default.yaml`, no rebuild needed)

Override any key with a dict or YAML file; it is deep-merged over the defaults, e.g.
`FinetuneDataset({"tasks": {"segmentation": {"class_list": "present"}}})`.

| Key | Default | Choices | Effect / when to change |
|---|---|---|---|
| `root` | `data/finetune_canonical` | path | where the built store is |
| `splits` | `[train]` | `train`, `val`, `test` | which split(s) to load (`test` only for analysis) |
| `seed` | `0` | int | default seed of `sample_indices` |
| `sources.<name>.enabled` | `true` | bool | drop a whole source |
| `sources.<name>.weight` | `1.0` | float | share of that source within each task it appears in |
| `sources.<name>.tasks` | all tasks it has; Hypersim: `[normals]` | list | which tasks a source is used for. Hypersim also has NYU-40 segmentation and depth: `tasks: [normals, segmentation, depth]` adds ~4,000 indoor synthetic images to both |
| `tasks.<task>.enabled` | `true` | bool | train without a task |
| `tasks.<task>.weight` | `1.0` | float | share of samples for that task (e.g. 2.0 doubles it) |
| `size.max_size` | `[1024, 1024]` | `[W, H]` | largest allowed output; larger images are scaled down |
| `size.max_area` | `null` | pixels | area cap |
| `size.upscale_long_side` | `null` | int | scale so the long side equals this (used for depth) |
| `size.size_multiple` | `32` | int | centre-crop to multiples of this (Qwen-Image needs 32) |
| `tasks.segmentation.size` | `{max_area: 524288}` | as above | as in the Cityscapes / ADE20K eval configs |
| `tasks.segmentation.class_list` | `all` | `all`, `present` | `all`: the prompt lists every class of the label space (40 for NYUv2, 10 for VKITTI2), as Vision Banana; `present`: only classes in the image, as RINO / Liu et al. See 7.2 |
| `tasks.segmentation.colour_format` | `name` | `name`, `rgb`, `hex` | how colours are written in the prompt ("purple" / "RGB(128, 0, 128)" / "#800080"). The eval configs use `name` |
| `tasks.segmentation.template` | eval prompt | str with `{class_color_list}` | custom prompt wording |
| `tasks.depth.size` | `{upscale_long_side: 1024}` | as above | as in `qwen_nyuv2_depth.yaml` |
| `tasks.depth.encoding` | `disparity` | `disparity`, `linear` | brightness linear in 1/depth (matches the main depth metric) or in depth |
| `tasks.depth.percentiles` | `[1, 99]` | `[lo, hi]` | per-image normalisation range |
| `tasks.depth.instruction` | eval prompt | str | custom prompt |
| `tasks.normals.size` | `{}` (native, max 1024) | as above | as in `qwen_normal_*.yaml` |
| `tasks.normals.instruction` | eval system prompt + prompt | str | custom prompt |
| `tasks.pose.canvas` | `[512, 768]` | `[W, H]` | RINO's fixed canvas; change only together with the pose eval config |
| `tasks.pose.joint_radius` | `6` | px | joint dot size in the target |
| `tasks.pose.bone_width` | `2` | px | bone line width; joints must stay thicker than bones (see 7.8) |
| `tasks.pose.instruction` | RINO's prompt | str | custom prompt |

If you change a prompt, size or encoding for training, make the same change in the evaluation config, or the
model will be tested on something it was not trained on.

### 4.2 Build options (change the data itself; rebuild that source, `verify` will then report differences)

| Source | Option | Default | Effect |
|---|---|---|---|
| `vkitti2` | `--test-scene` | `Scene02` | which scene is held out as test (Scene02 is the shortest) |
| `vkitti2` | `--stride` | `5` | keep every n-th frame (multiples of 5: only frames ending in 0 / 5 are downloaded) |
| `vkitti2` | `--variations` | 6 weather / time variations | e.g. `clone fog` |
| `vkitti2` / `nyuv2` | `--val-count`, `--seed` | `50`, `0` | size and draw of the val split |
| `hypersim` | `--train-frames` | `4000` | spread evenly over the 347 training scenes (3965 after scenes with fewer frames) |
| `coco_pose` | `--train-count`, `--val-count`, `--seed` | `4000`, `100`, `0` | one person per image, random with the seed |

## 5. Targets and prompts

| Task | Target image | Instruction | Decoded by |
|---|---|---|---|
| Segmentation | each class painted in its flat colour from `palette.json`, everything else (void / unlabelled) black | `Convert this photo into a color-coded map: the wall purple, the floor ..., and everything else black.` | `tasks/evaluators/semseg.py` (nearest prompted colour) |
| Depth | grayscale, near = white, brightness linear in 1/depth, normalised per image to the 1st–99th percentile; invalid pixels (no depth, VKITTI2 sky) black | the RINO depth prompt (`nyuv2_depth.py`) | `tasks/evaluators/nyuv2_depth.py` (luminance, affine fit) |
| Normals | R = (1 − x)/2, G = (1 + y)/2, B = (1 + z)/2 in the evaluation GT frame (x right, y up, z towards the camera); invalid pixels black | kiki's system prompt + normals prompt | `tasks/evaluators/surface_normal.py` with `rgb_to_gt: [-r, +g, +b]` |
| Pose | OpenPose-coloured skeleton on black (COCO-17 joints as dots, bones as lines), on the 512 × 768 canvas | RINO's pose prompt | `tasks/evaluators/pose_skeleton.py` |

`palette.json` is one global class-to-colour table: the same class has the same colour in every dataset (NYUv2
"person" and Cityscapes "person" are both blue), a colour name always means one RGB value, and Cityscapes
classes keep the exact colours of the Cityscapes evaluator. Colours inside one label space are chosen to be as
far apart as possible (minimum CIELAB distance 23 for NYUv2-40, 29 for VKITTI2). Regenerate with
`python -m finetune_data.palette` (only if classes change; it changes the training targets).

`tests/test_finetune_data.py` decodes the targets with the real evaluators and checks they score perfectly.

## 6. Evaluation

### 6.1 Test configs

All are run with `uv run visionbench run --config configs/<name>.yaml`. On AWS (DLAMI) prefix
`env -u LD_LIBRARY_PATH` (the DLAMI CUDA libraries break cuDNN).

| Config | Test set | Notes |
|---|---|---|
| `indomain_nyuv2_semseg_present.yaml` / `_all.yaml` | NYUv2 test, 654 images, 40 classes | read from the store with the training renderer's sizes, prompts and colours |
| `indomain_vkitti2_semseg_present.yaml` / `_all.yaml` | VKITTI2 Scene02, 282 images, 10 classes | same |
| `indomain_vkitti2_depth.yaml` | VKITTI2 Scene02, 282 images | `max_depth: 80` (KITTI convention), no Eigen crop |
| `qwen_nyuv2_depth.yaml` (Ahmad's) | NYUv2 test, 654 images | same images as the store, but raw sensor depth (Marigold protocol), see 7.5 |
| `diode_depth.yaml` | DIODE val, 771 images (out-of-domain) | Marigold packaging, 0.6–350 m; data: `scripts/download_diode_depth.sh` (6.4 GB) |
| `full_qwen_cityscapes_{present,all}.yaml`, `full_qwen_ade20k.yaml` | Cityscapes val 500, ADE20K val 2,000 (out-of-domain) | never trained on |
| `qwen_normal_{nyuv2,ibims,diode}.yaml` (kiki's) | normals test sets (out-of-domain) | |
| `coco_pose_estimation.yaml` | COCO val2017, 3,061 persons | RINO protocol; reports `rino/*` and `colour/*` (see 7.8); data: `scripts/download_coco_pose.sh` |
| `coco_pose_qwen.yaml` (lc660's) | COCO val2017, 1,593 images | pose *preservation*: a check of editing ability, not a training task (7.9) |

To evaluate a fine-tuned model: copy the config, set a new `generation_version` (outputs and resume state are
keyed by it), and point the model at the checkpoint. **The Qwen backend (`visionbench/models/qwen.py`) cannot
load LoRA weights yet**; that needs to be added (e.g. `pipe.load_lora_weights(...)` behind a new model option)
before fine-tuned checkpoints can be scored.

### 6.2 Zero-shot baselines (Qwen-Image 2.1, 40 steps, bf16, seed 42)

| Test set | Metric | Zero-shot | Source |
|---|---|---|---|
| NYUv2 seg, present / all | mIoU | 32.8 / 7.2 | `outputs/nyuv2_semseg/` |
| VKITTI2 seg, present / all | mIoU | 51.6 / 48.4 | `outputs/vkitti2_semseg/` |
| VKITTI2 depth | AbsRel / δ1 | 0.281 / 0.470 | `outputs/vkitti2_depth/` |
| NYUv2 depth | AbsRel / δ1 | 0.0845 / 0.922 | midterm report, Table 1 |
| Cityscapes, present / all | mIoU | 42.7 / 30.5 | midterm report, Table 2 |
| ADE20K, present | mIoU | 37.5 | midterm report, Table 2 |
| Normals NYUv2 / iBims / DIODE-indoor | mean angle | 19.07 / 18.80 / 18.14 | midterm report, Table 3 |
| Pose preservation | PCK@0.1 | 0.898 | midterm report, Table 4 |
| DIODE depth | AbsRel / δ1 | not run yet (~7.5 h on a g5.2xlarge) | `diode_depth.yaml` |
| COCO pose estimation | PCK@0.1 | not run yet (~22 h full; ~7 h with `sample_stride: 3`) | `coco_pose_estimation.yaml` |

The two missing baselines only need GPU time; both configs were run end to end with the real model. With
`sample_stride: 3`, use the same stride for the fine-tuned model.

### 6.3 Ceilings: what a perfect prediction scores

Scored with the ground-truth targets as predictions:

| Test set | Ceiling | Why |
|---|---|---|
| NYUv2 seg / depth | 100% / ~0 AbsRel | evaluated at the model resolution |
| VKITTI2 seg / depth | mIoU 0.94 / AbsRel 0.03 | the model sees 1024 px wide images, scoring is at the original 1242 px; all losses lie within 2 px of boundaries |
| COCO pose, `colour/*` | PCK@0.1 ≈ 0.99 | |
| COCO pose, `rino/*` | PCK@0.1 ≈ 0.37–0.44 | RINO's decoder, see 7.8 |

## 7. Decisions and known issues

Each item: the problem, what we do now, and what to watch.

**7.1 Out-of-domain sets are never trained on.** Cityscapes, ADE20K, DIODE and the normals test sets are not
in the training data (the midterm plan listed Cityscapes / ADE20K training sets for Direction 3; we left them
out). Training on them would make the "out-of-domain" numbers in-domain and break the comparison with the
zero-shot baseline and Vision Banana (which reports Cityscapes without Cityscapes training data). There are no
NYUv2 training normals with GT in the Marigold format, so normals are trained on Hypersim.

**7.2 Listing all classes vs. only present classes.** Zero-shot, NYUv2 scores 32.8 mIoU with present classes
but 7.2 with all 40. The outlines are right in both; with 40 colours the model cannot paint the exact shade
(e.g. wall should be purple (128, 0, 128), it paints a lighter purple, which is then nearest to another class).
Training uses `class_list: all` (the realistic setting: at test time we do not know which classes are present).
Watch: this colour precision is where fine-tuning should help most; report both protocols, and keep
`colour_format` the same in training and testing.

**7.3 Cityscapes' pink and light pink.** In the Cityscapes evaluation palette (from RINO), traffic sign is pink
(255, 192, 203) and bicycle is light pink (255, 182, 193): almost the same colour. In the zero-shot outputs
more traffic-sign pixels are decoded as bicycle (9.3%) than as traffic sign (4.3%). We keep the palette so the
numbers stay comparable with RINO and our earlier results; the two classes' IoU (and a little of the mIoU) are
lower for every model. State this as a limitation in the report. Our own label spaces avoid the problem: the
VKITTI2 label space has traffic sign but no bicycle.

**7.4 Depth direction.** Targets are always near = white. Zero-shot, Qwen-Image paints VKITTI2 the other way
round in 76% of images (median correlation with 1/depth: −0.80). The depth evaluator's least-squares fit
accepts a negative scale, so an inverted map is not punished as such (which is why the NYUv2 runs never showed
it), but it still fits worse, hence VKITTI2's low zero-shot score. Expect a large gain here after fine-tuning.

**7.5 NYUv2 depth: training vs. testing depth.** The training store uses the official in-painted (hole-filled)
NYUv2 depth; the test config `qwen_nyuv2_depth.yaml` keeps the raw sensor depth (Marigold protocol), so results
stay comparable with the report.

**7.6 VKITTI2 specifics.** Sky (stored by VKITTI2 as 655.35 m) is marked invalid: black in the target and
excluded in testing; use `max_depth: 80` for VKITTI2 depth. The test scene is Scene02 (a whole scene, so near-
identical neighbouring frames cannot leak). Classes are mapped to Cityscapes names and colours (Tree and
Vegetation -> vegetation, Van -> car, GuardRail / Misc -> void).

**7.7 Hypersim normals frame.** We follow Marigold's preprocessing (tone-mapped HDR image, unit normals,
back-facing normals flipped), so the normals are in exactly the frame of the Marigold evaluation GT (checked
against Marigold's NYUv2 GT). Our target encoding matches kiki's calibrated `rgb_to_gt: [-r, +g, +b]`.
Watch: if the evaluator's `rgb_to_gt` is ever re-calibrated, the training encoding in `encoders.normals` must
change with it.

**7.8 Pose: RINO's decoder has a low ceiling.** The pose task follows RINO (person crop, 512 × 768 canvas,
OpenPose skeleton, RINO's prompt and metrics; code ported under MIT in `tasks/evaluators/_rino_pose.py`).
RINO's decoder ignores colour and fits detected dots to an upright, front-facing template, so even perfectly
drawn GT skeletons score only ~0.37 PCK@0.1 on COCO val (side views and bent poses get mislabelled). We added a
decoder that identifies joints by their OpenPose colour (ceiling ~0.99). The evaluator reports both: use
`rino/*` to compare with RINO's numbers, `colour/*` for what a fine-tuned model actually draws. The target's
joint radius 6 / bone width 2 were chosen because both decoders recover the most joints at these sizes.

**7.9 Two different "pose" tasks.** `coco_pose_estimation.yaml` (this data) asks for a skeleton. lc660's
`coco_pose_qwen.yaml` asks the editor to keep the photo and pose unchanged and checks it with Keypoint R-CNN;
its training pairs would be photo -> same photo, so it is not a training task. Keep it as a check that
fine-tuning does not damage editing.

**7.10 Resolution.** Each task keeps its evaluation preprocessing (section 2). VKITTI2 is scored at its
original 1242 px width, which caps a perfect 1024 px prediction at 0.94 mIoU (section 6.3).

**7.11 Splits never overlap.** NYUv2 uses the official train / test split; VKITTI2 holds out a whole scene;
COCO trains on train2017 and tests on val2017; Hypersim uses Marigold's training scenes only.

**7.12 iBims-1 depth is not scripted.** Its official download is a password-protected TUM share, so DIODE is
the out-of-domain depth set. iBims-1 normals (kiki's config) are unaffected.

**7.13 Pose has no out-of-domain test yet.** Training is COCO train2017 and testing COCO val2017, so pose
results show in-domain gains only. Candidates with the same COCO-17 keypoints (they plug into the same RINO
protocol by changing the loader's `annotations_file` / `images_dir`): OCHuman (heavily occluded people, public,
val / test only, the usual "train on COCO, test on OCHuman" setting) and Human-Art (paintings, cartoons,
sculptures; non-commercial, needs an access request form). Not added yet.

**7.14 Download sources.** VKITTI2's server resets long connections; the download script resumes and retries.
If a source ever disappears, the data can no longer be rebuilt from scratch; ask Lilian for a copy.

## 8. Reproducibility and file layout

`scripts/build_finetune_data.sh` downloads and builds every source (skipping those already built), then runs
`finetune_data.verify`, which compares decoded content hashes (pixels, float16 normals, canonical JSON rows)
with `finetune_data/checksums/`. Different PNG encoders or zlib versions do not matter; any change in the data
is reported per file. Builds were checked to be identical on macOS arm64 and Linux x86_64. Every random choice
is seeded and every external file list (Marigold, Hypersim) is pinned to a commit.

Per-source builds, if you only need some of it:

```bash
bash scripts/download_nyuv2_train.sh                                   # 2.8 GB -> raw/nyuv2_train/
uv run python -m finetune_data.sources.nyuv2 raw/nyuv2_train data/finetune_canonical
bash scripts/download_vkitti2.sh                                       # ~17 GB downloaded, ~1 GB kept
uv run python -m finetune_data.sources.vkitti2 raw/vkitti2 data/finetune_canonical
uv run python -m finetune_data.sources.hypersim data/finetune_canonical --train-frames 4000   # streamed, ~8.6 GB kept
uv run python -m finetune_data.sources.coco_pose raw/coco2017 data/finetune_canonical          # ~0.7 GB
```

The store (`data/finetune_canonical/<source>/`) keeps original-resolution data; rendering happens later:

| File | Content |
|---|---|
| `meta.json` | source, label space, depth units, split sizes, build parameters |
| `manifest.jsonl` | one row per image: `id, source, split, image` and one entry per annotation |
| `images/<id>.png` | RGB input at original resolution (COCO: the person crop) |
| `seg/<id>.png` | uint8 class ids in the source's label space (0 = void) |
| `depth/<id>.png` | uint16 depth; `meta.depth_units_per_m` units per metre (1000 = mm, VKITTI2: 100 = cm); 0 = invalid |
| `normals/<id>.npz` | float16 H × W × 3 unit normals in the evaluation frame; NaN = invalid |
| `pose` (in the manifest row) | `keypoints`: 17 × (x, y, v) in crop pixels (COCO visibility), `bbox_wh`: tight box size |

`data/` and `raw/` are gitignored; never commit them.

## 9. Code map

| Path | Purpose |
|---|---|
| `finetune_data/dataset.py` | `FinetuneDataset`, `subset_config`, balanced sampling, rendering |
| `finetune_data/configs/default.yaml` | all rendering options (section 4.1) |
| `finetune_data/encoders.py`, `prompts.py`, `palette.py` / `palette.json` | target encodings, instructions, colours |
| `finetune_data/sources/{nyuv2,vkitti2,hypersim,coco_pose}.py` | builders of the canonical store |
| `finetune_data/export.py` | write triplets as files (`--tasks`, `--sources`, `--per-group`) |
| `finetune_data/verify.py`, `checksums/` | reproducibility check |
| `finetune_data/remote_zip.py` | read single files from remote zips (Hypersim, COCO annotations) |
| `scripts/build_finetune_data.sh`, `scripts/download_*.sh` | one-command build; downloads |
| `tasks/dataloaders/finetune_{semseg,depth}.py`, `_finetune_store.py` | in-domain test loaders (read the store) |
| `tasks/dataloaders/coco_pose_estimation.py`, `tasks/evaluators/{_rino_pose,pose_skeleton}.py`, `tasks/aggregators/pose_skeleton.py` | RINO pose estimation evaluation |
| `tasks/dataloaders/diode_depth.py` | DIODE depth loader |
| `tests/test_finetune_data.py` | 19 tests: targets decode perfectly with the evaluators, sizes, sampling, splits, sources, verification |
