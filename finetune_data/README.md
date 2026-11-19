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

```bash
bash scripts/download_nyuv2_train.sh                                   # ~2.8 GB -> raw/nyuv2_train/
uv run python -m finetune_data.sources.nyuv2 raw/nyuv2_train data/finetune_canonical
```

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

## Sources

| Source | Tasks | Splits | Notes |
|---|---|---|---|
| `nyuv2` | segmentation (NYU-40), depth | 745 train / 50 val / 654 test | official labeled set; test = the Eigen split our evaluators use |

Not yet added: Virtual KITTI 2 (segmentation, depth), Hypersim (normals), COCO train2017 (pose).
Cityscapes and ADE20K train splits are deliberately excluded so segmentation results stay
comparable to the zero-shot baseline and Vision Banana.
