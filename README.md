# VisionBench

Stage 1 image-to-image benchmarking for vision tasks. A run first saves every generated image, then evaluates successful outputs. Teammates only write a dataset loader, evaluator, and optional aggregator. [SPEC.md](SPEC.md) records the agreed behavior and acceptance criteria.

## Quick start

On Linux with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --frozen
cp .env.example .env
uv run visionbench run --config config_dummy
```

The dummy run needs no GPU or API key. It uses four small images and scores the fraction of exact white pixels. The default real-model configuration is [`configs/qwen.yaml`](configs/qwen.yaml), which uses a 16 GB-oriented NF4/offload profile and a pinned Qwen-Image 2.1 revision. Run it after confirming a CUDA driver and sufficient model-cache storage:

```bash
uv run visionbench run --config qwen
```

The first Qwen run downloads a large checkpoint. Set `model.cache_dir` to persistent storage. Image dimensions affect peak memory: the 16 GB profile is a starting configuration, not a guarantee for arbitrary input sizes. The runner never resizes input images or silently changes quantization. If the GPU runs out of memory, it stops; preprocess in the dataset loader and use a new `generation_version` when changing the generated input.

To test one image without running the dataset:

```bash
uv run visionbench infer --config config_dummy --image raw/white_background/sample_01.png --prompt 'Change the background to white #FFFFFF'
```

Available commands are `run`, `generate`, `evaluate`, `infer`, and `archive`. `--config` accepts a name from `configs/` (`qwen`, `ark`, `config_dummy`, and so on) or an explicit YAML path. `run` executes generation, then evaluation, then optional S3 archival. `generate` resumes completed outputs by default. `evaluate` starts a new numbered evaluation every time. `infer` writes a debug PNG and JSON outside benchmark runs. Open [`notebooks/01_single_sample_debug.ipynb`](notebooks/01_single_sample_debug.ipynb) or [`notebooks/02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb) from the repository root for interactive work.

## Task code

Each task uses three separate files under [`tasks/`](tasks/): [`tasks/dataloaders/`](tasks/dataloaders/), [`tasks/evaluators/`](tasks/evaluators/), and [`tasks/aggregators/`](tasks/aggregators/). Put unmodified source dataset files under [`raw/`](raw/) and optional task support files under [`assets/`](assets/). The toy input images are in [`raw/white_background/`](raw/white_background/). Copy the examples in [`tasks/dataloaders/white_background.py`](tasks/dataloaders/white_background.py), [`tasks/evaluators/white-background.py`](tasks/evaluators/white-background.py), and [`tasks/aggregators/white_background.py`](tasks/aggregators/white_background.py).

YAML paths are relative to their dedicated folder. All supplied YAML files, including [`configs/config_dummy.yaml`](configs/config_dummy.yaml), live in `configs/`. For example:

```yaml
dataset:
  path: white_background.py   # tasks/dataloaders/white_background.py
  raw_path: white_background  # raw/white_background/
  assets_path: .              # assets/ (optional; default)
  params:
    max_size: [1024, 1024]
    shrink_mode: scale
    size_multiple: 16
evaluator:
  path: white-background.py   # tasks/evaluators/white-background.py
aggregator:
  path: white_background.py  # tasks/aggregators/white_background.py
```

The runner passes resolved `raw_dir` and `assets_dir` strings to `load_dataset(config)`, alongside `dataset.params`. Absolute paths, `..` escapes, and symlinks that leave a dedicated folder are rejected for these YAML paths. Keep source inputs in `raw/`; the runner's `outputs/.../inputs/` folder stores the processed images actually sent to a model.

```python
# tasks/dataloaders/my_task.py
from visionbench.types import Sample

class MyDataset:
    def __len__(self): ...
    def __getitem__(self, index) -> Sample: ...  # stable ID, RGB/RGBA Pillow image, prompt, arbitrary GT

def load_dataset(config: dict) -> MyDataset: ...
```

```python
# tasks/evaluators/my_task.py
METRICS = {"score_name": {"unit": "fraction", "higher_is_better": True}}

def prepare_output(image, sample, config):  # optional
    return image

def score(generated_image, sample, config) -> dict[str, float]: ...
```

```python
# tasks/aggregators/my_task.py; omit this file/path to use a mean across samples
def aggregate(records: list[dict], config: dict) -> dict[str, float]: ...
```

`records` contains a lazy `sample` reference (`.load()` retrieves the full `Sample`; fields such as `.ground_truth` can be accessed directly), its trial-mean `metrics`, trial records, output paths, and failure count. Keeping samples lazy prevents thousands of large images from accumulating in RAM. Every trial is scored independently; its metrics are averaged within the sample before aggregation. Invalid or missing metric values are warned about and excluded with per-metric coverage reported. Scoring exceptions affect only the relevant trial. Dataset loading errors and duplicate IDs stop the run during full preflight, before model loading or paid calls.

The sample loader calls `visionbench.transforms.preprocess_image(image, max_size=(1024, 1024), shrink_mode="scale", size_multiple=16)` and stores the serializable transform and parameters in sample metadata. The helper first shrinks if needed, then center-crops down to the multiple. It supports forward/inverse points and boxes, visibility checks, and mask/depth/normal raster transforms. The runner itself never changes image geometry. The Qwen backend currently requires dimensions divisible by **32**, so `configs/qwen.yaml` sets `size_multiple: 32`; set the same preprocessing parameters in every model config when comparing them on a new dataset.

## NYUv2 depth estimation

[`configs/qwen_nyuv2_depth.yaml`](configs/qwen_nyuv2_depth.yaml) benchmarks zero-shot monocular depth on the NYUv2 Eigen test split (654 images) with the protocol from RINO ([Let RGB Be the Language of Vision](https://arxiv.org/abs/2607.12450)). The model is asked to repaint the photo as a grayscale depth visualization (near = bright, far = dark). [`tasks/evaluators/nyuv2_depth.py`](tasks/evaluators/nyuv2_depth.py) reads BT.709 luminance as a disparity-like relative map. It maps the map back to the 640×480 ground-truth frame and keeps raw sensor depth in `(1e-3, 10)` m inside the Eigen crop. It then fits a per-image least-squares scale and shift before scoring.

```bash
bash scripts/download_nyuv2.sh   # Marigold's packaged test split -> raw/nyuv2/ (~1.1 GB download)
uv run visionbench run --config qwen_nyuv2_depth
```

The headline metrics (`abs_rel`, `delta1`, ...) align in disparity space (`s·luma + t ≈ 1/gt`), which is the MiDaS / Depth Anything convention and RINO Table 1. The `*_depthspace` metrics align in linear depth (`s·luma + t ≈ gt`), the Marigold convention and RINO Table 2. Least squares may choose a negative scale, so polarity is recovered and only depth ordering and shape are scored. The loader upscales 640×480 inputs to 1024×768 (`upscale_long_side`) so Qwen-Image 2.1 generates near its ~1 MP condition resolution. Set `limit: N` under `dataset.params` together with a new `generation_version` for a quick subset run. The supplied model profile keeps everything in bf16 on one 48 GB GPU (about 27 s per image at 40 steps on an RTX A6000). Use `profile: vram_16gb` on smaller cards.

Zero-shot results on all 654 NYUv2 test images (one trial, seed 42, 40 steps, 1024×768):

| Model | AbsRel ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | RMSE ↓ | AbsRel (depth-space) ↓ | δ1 (depth-space) ↑ |
|---|---|---|---|---|---|---|---|
| Qwen-Image 2.1 (`790c926`), this repo | 0.0845 | 0.922 | 0.982 | 0.995 | 0.385 | 0.0987 | 0.906 |
| Qwen-Image-Edit, RINO Table 1 | 0.086 | 0.927 | – | – | – | – | – |
| Depth Anything V2, RINO Table 1 | 0.045 | 0.979 | – | – | – | – | – |

## Semantic segmentation (Cityscapes, ADE20K)

`configs/full_qwen_{cityscapes_all,cityscapes_present,ade20k}.yaml` benchmark zero-shot semantic segmentation on Cityscapes val (500 images, 19 classes) and ADE20K val (2,000 images, 150 classes) with the protocol from RINO ([Let RGB Be the Language of Vision](https://arxiv.org/abs/2607.12450)). The prompt assigns a flat colour to each class (`Convert this photo into a color-coded map: the road red, …, and everything else black.`). [`tasks/evaluators/semseg.py`](tasks/evaluators/semseg.py) maps each pixel to the nearest prompted colour (black = no class) and scores at the original ground-truth resolution. [`tasks/aggregators/semseg.py`](tasks/aggregators/semseg.py) computes mIoU, mAcc and aAcc from pixel counts summed over the split, with 95% bootstrap intervals, per-class IoU, RINO's refinement (`*_refined`) and coarse-group scores (`coarse_fine_*`).

```bash
bash scripts/download_cityscapes.sh   # needs a free cityscapes-dataset.com account (~11 GB)
bash scripts/download_ade20k.sh       # ~920 MB, no login
uv run visionbench run --config full_qwen_cityscapes_all       # likewise full_qwen_cityscapes_present, full_qwen_ade20k
```

`class_list: present` names only the classes in each image's ground truth, as in RINO and Liu et al. ([Open-Source Image Editing Models Are Zero-Shot Vision Learners](https://arxiv.org/abs/2605.04566)). `class_list: all` names all 19 Cityscapes classes, as in Vision Banana ([Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329)). It does not leak labels and is the fine-tuning baseline. ADE20K uses `present`. RINO's 150-colour ADE20K palette pairs colour names with unrelated RGB values (its "orange" is (29, 58, 127)). `palette: named` uses colours whose names match their RGB, assigned by dataset-wide class frequency. RINO's palette and two-pass mode remain available. "refined" applies RINO's post-processing (5×5 mode filter, blobs under 100 px removed) to the same outputs. A ground-truth-painted prediction scores 100%, and decoding matches RINO's code pixel for pixel. `tasks/subsets/` holds fixed 100-image subsets for quick runs (`subset_file:`). The configs fit a 24 GB A10G with the text encoder CPU-offloaded. `max_pixels` scales down the two ADE20K images that otherwise run out of memory. On AWS Deep Learning AMIs, run with `env -u LD_LIBRARY_PATH`.

Results on the full validation splits (one trial, seed 42, 40 steps; one A10G, ~33 s per Cityscapes image and ~18 s per ADE20K image). mIoU 95% intervals: Cityscapes all 29.6–31.3, present 40.6–45.2; ADE20K 36.0–39.4.

Cityscapes val (500 images):

| Model | Training | Protocol | mIoU ↑ | mAcc ↑ | aAcc ↑ | 7-group mIoU ↑ |
|---|---|---|---|---|---|---|
| Qwen-Image 2.1 (`790c926`), this repo | none (zero-shot editor) | present | 42.7 | 56.6 | 87.8 | 71.7 |
| Qwen-Image 2.1, this repo, refined | none (zero-shot editor) | present | 43.3 | 56.4 | 88.1 | 72.1 |
| Qwen-Image 2.1, this repo | none (zero-shot editor) | all | 30.5 | 45.6 | 80.9 | 63.6 |
| Qwen-Image 2.1, this repo, refined | none (zero-shot editor) | all | 30.9 | 45.7 | 81.3 | 64.3 |
| Qwen-Image-Edit, Liu et al. Table 4 | none (zero-shot editor) | present | 25.7 | – | 55.4 | 49.5 |
| Qwen-Image-Edit, RINO Table 4 | none (zero-shot editor) | present, refined | 24.3 | 44.4 | 66.6 | – |
| FC-CLIP, RINO Table 4 | segmentation training on other data, no Cityscapes | – | 56.3 | 65.4 | 78.5 | – |
| SAM 3, Vision Banana Table 2 | segmentation training on other data, no Cityscapes | – | 65.2 | – | – | – |
| Vision Banana, Table 2 | vision instruction tuning, no Cityscapes | all | 69.9 | – | – | – |
| SegMan-L, Vision Banana Table 2 | trained on Cityscapes (upper reference) | – | 84.2 | – | – | – |

ADE20K val (2,000 images):

| Model | Training | Palette | 150-class mIoU ↑ | mAcc ↑ | aAcc ↑ | 10-group mIoU ↑ | mAcc ↑ | aAcc ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen-Image 2.1 (`790c926`), this repo | none (zero-shot editor) | named | 37.5 | 53.4 | 67.6 | 55.6 | 69.6 | 72.9 |
| Qwen-Image 2.1, this repo, refined | none (zero-shot editor) | named | 38.7 | 54.5 | 70.3 | 56.9 | 71.6 | 76.7 |
| Qwen-Image-Edit, RINO Table 5 | none (zero-shot editor) | RINO | 12.6 | 20.9 | 44.9 | 40.4 | 53.5 | 67.4 |
| FC-CLIP, RINO Table 5 | segmentation training on other data, no ADE20K | – | 34.1 | 40.9 | 59.6 | 42.2 | 53.1 | 67.1 |
| MaskCLIP, RINO Table 5 | CLIP-based, no ADE20K training | – | 23.8 | 31.5 | 50.3 | 39.3 | 50.5 | 65.7 |

Liu et al. report pixel accuracy, listed under aAcc. Our 10-group scores collapse the 150-class prediction; RINO prompts a separate 10-group pass. FC-CLIP and MaskCLIP do not receive the per-image class list. No generation or scoring failures occurred.

## YAML and credentials

The YAML files are [`configs/config_dummy.yaml`](configs/config_dummy.yaml), [`configs/qwen.yaml`](configs/qwen.yaml), [`configs/openai.yaml`](configs/openai.yaml), [`configs/gemini.yaml`](configs/gemini.yaml), and [`configs/ark.yaml`](configs/ark.yaml). They select explicit model IDs, a multiline system prompt, task files/parameters, trial count, remote concurrency (default 3), retry policy, output root, and S3 archive settings. Local Qwen always runs one generation at a time. Model-specific controls belong under `model.options`; invalid Qwen combinations fail before inference. Model-specific prompt tuning should use a separately labeled run; all main comparisons should use the same task and sample prompts.

Copy `.env.example` to `.env` for API keys. YAML uses `${OPENAI_API_KEY}`, `${GEMINI_API_KEY}`, or `${ARK_API_KEY}`; literal API keys are rejected. `.env` is ignored by Git, and saved resolved configurations redact secrets. OpenAI uses the Images edit API, Google uses the Gemini Developer API, and Seedream uses ByteDance Volcano Ark. The adapters accept provider-native options, and API responses may return a different output size. Task `prepare_output` handles alignment before scoring. Provider API availability, access, and billing must be checked in each account.

The original model list said Seedance; `configs/ark.yaml` selects Seedream 5.0 Pro for this image task. Confirm that substitution before using its scores in a comparison. The four tiny example images are for the dummy smoke test; for real providers, use a dataset whose sizes satisfy each provider's current input and output limits.

For S3, set `s3.enabled: true`, `s3.bucket`, and optional `s3.prefix` in YAML. Use an EC2 IAM role or standard `AWS_*` environment variables. The archive command uploads a tarball of the local run. S3 is only an archive; local checkpoints drive resumption. Archive failure is logged without deleting local results.

## Run layout and resume

```text
outputs/{task_name}/{generation_version}/
  manifest.json
  config.resolved.json
  run.log
  inputs/                         # processed images received from the loader
  images/                         # raw generated images, one per trial
  checkpoints/                    # one atomic record per generation trial
  generation_summary.json
  evaluations/0001/results.json
  evaluations/0002/results.json
```

The manifest fingerprints every processed input image and prompt. A rerun with unchanged generation settings skips completed outputs and retries failed ones. Changing a model, generation setting, prompt, or input image under the same `generation_version` raises an error; choose a new version. Changing evaluator or aggregator code can reuse images and creates the next evaluation number. Evaluation restarts from the beginning when rerun. Each `results.json` contains scores, full per-sample/per-trial records, failures, timing, model metadata, and coverage.

## Docker on AWS or another Linux GPU host

Install the NVIDIA driver and NVIDIA Container Toolkit on the host, then mount this repository and a persistent model cache:

```bash
docker build -t visionbench .
docker run --rm --gpus all --env-file .env \
  -v "$PWD/outputs:/app/outputs" -v "$PWD/model-cache:/app/model-cache" \
  -v "$PWD/raw:/app/raw:ro" -v "$PWD/assets:/app/assets:ro" \
  visionbench run --config qwen
```

The image installs all model and API dependencies. It includes only the four toy raw images; mount `raw/` and `assets/` to use larger datasets without baking them into the image. For API-only calls, the same image can run without `--gpus all`. On EC2, ensure the instance has enough EBS space for model weights, datasets, and saved inputs/outputs; an instance profile is the easiest way to grant S3 archive permissions. Model download and real GPU/API generation were not available in the development environment, so use the dummy run and notebooks first, then manually smoke-test one image per real provider and Qwen profile.

## COCO pose benchmark

Adds a COCO human pose preservation benchmark task for the POS metrics experiment. The task follows a fixed pose-estimation protocol: use COCO Keypoints 2017 validation images, ask the image editor to preserve the person and pose, then run a fixed pretrained pose estimator on the generated image and compare predicted keypoints against the original COCO ground truth.

- `tasks/dataloaders/coco_keypoints.py`: loads COCO 2017 `val2017` images with `annotations/person_keypoints_val2017.json`. The first version selects one visible dominant person per image, filters for enough visible keypoints and minimum person area, preprocesses images to Qwen-compatible dimensions, and transforms ground-truth keypoints and boxes into the processed image frame.
- `tasks/evaluators/coco_pose.py`:
  - Runs a fixed torchvision Keypoint R-CNN pose detector on each generated image.
  - Matches the predicted person to the ground-truth person by bounding-box IoU.
  - Scores visible COCO keypoints with normalized keypoint error.
  - Reports PCK@0.05, PCK@0.10, PCK@0.20, NMKE, pose detection failure rate, missing-keypoint rate, and per-image COCO-style OKS.
- `configs/coco_pose_qwen.yaml`: Qwen-Image 2.1 config for the COCO pose task.
- `tests/test_coco_pose_loader.py`: loader selection, preprocessing, box transform, and keypoint transform coverage.
- `scripts/download_coco_pose.sh`: downloads and extracts the required COCO files into the ignored `raw/coco2017/` directory. This task has no assets to download.

### Dataset setup

```bash
./scripts/download_coco_pose.sh
```

The script fetches the 5,000-image COCO 2017 validation split and
`person_keypoints_val2017.json`, then creates the layout expected by the
config:

```text
raw/coco2017/
├── val2017/
└── annotations/person_keypoints_val2017.json
```

`raw/` and `assets/` are gitignored. Downloaded datasets must not be committed.
Set `RAW_ROOT=/path/to/storage/raw` to put the dataset on a different volume.

### Results (complete filtered COCO split, RTX 4090)

| Model | PCK@0.05 ↑ | PCK@0.10 ↑ | PCK@0.20 ↑ | NMKE ↓ | Detection failure ↓ | Missing keypoint ↓ | OKS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-Image 2.1, complete 1,593-sample COCO pose run | 0.8379 | 0.8981 | 0.9275 | 0.0611 | 0.0320 | 0.0536 | 0.7869 |

The evaluation begins with the 5,000-image COCO validation set and retains
1,593 unambiguous, sufficiently large people (at least 10 visible keypoints)
under the loader's fixed filtering protocol. All 1,593 generations succeeded,
with 0 scoring failures. Qwen-Image 2.1 was pinned to revision
`790c92633540aa0cb11d9abf19eb46d861714758`, run with 40 steps on one RTX 4090.

## Surface normal estimation

`configs/qwen_normal_{nyuv2,ibims,diode}.yaml` benchmark surface normals on NYUv2 test (654), iBims-1 (100) and DIODE-indoor (325) with the datasets of RINO ([Let RGB Be the Language of Vision](https://arxiv.org/abs/2607.12450)) Table 3. Splits, GT normals and valid masks come from the Marigold normals release, the same source as the DSINE / Marigold / StableNormal baselines in that table. The model repaints the photo as an RGB normal map. [`tasks/evaluators/surface_normal.py`](tasks/evaluators/surface_normal.py) decodes `v = 2·RGB/255 − 1`, resizes bilinearly back to the GT frame, renormalizes, applies a fixed axis convention, and scores the per-pixel angle. It reports mean and median angular error and the % of pixels under 11.25°, 22.5° and 30°, per image and then averaged over images.

```bash
bash scripts/download_surface_normal.sh   # Marigold normals eval zip -> raw/surface_normal/ (~21 GB download, resumable)
uv run visionbench generate --config qwen_normal_calib                                  # 50 DIODE-outdoor images
uv run python tasks/evaluators/surface_normal.py calibrate --config qwen_normal_calib   # prints rgb_to_gt
uv run visionbench run --config qwen_normal_ibims    # likewise qwen_normal_diode, qwen_normal_nyuv2
```

Generated normal maps use an unknown axis and sign convention. `calibrate` scores all 48 signed axis permutations on DIODE-outdoor images, which are disjoint from every test set, and prints the best one. Set it once as `evaluator.params.rgb_to_gt` in the test configs; it is `["-r", "+g", "+b"]` for Qwen-Image 2.1 (24.6° vs 51.3° for the runner-up), so re-calibrate for each new model. Only zero-length GT vectors are masked. Predictions with no direction inside valid GT count as 90° and are reported as `degenerate_pixel_pct`, and failed generations count as 90° for the whole image (`failed_sample_policy: penalize`). `*_pilot.yaml` configs run uniform subsets for a quick check.

Results on the full test sets (one trial, seed 42, 40 steps, native resolution; one A40, ~16 s per 640×480 image and ~33 s per 1024×768 image):

| Dataset | Model | Mean ↓ | Median ↓ | < 11.25° ↑ | < 22.5° ↑ | < 30° ↑ |
|---|---|---|---|---|---|---|
| NYUv2 | Qwen-Image 2.1 (`790c926`), this repo | 19.07 | 11.66 | 50.74 | 74.77 | 81.52 |
| | Qwen-Image-Edit, RINO Table 3 | 20.21 | 13.46 | – | – | – |
| | StableNormal, RINO Table 3 | 19.71 | 10.53 | 53.04 | 75.89 | 81.72 |
| iBims-1 | Qwen-Image 2.1 (`790c926`), this repo | 18.80 | 9.98 | 58.11 | 79.17 | 83.16 |
| | Qwen-Image-Edit, RINO Table 3 | 20.66 | 12.03 | – | – | – |
| | StableNormal, RINO Table 3 | 17.25 | 8.06 | 66.66 | 81.13 | 84.63 |
| DIODE-indoor | Qwen-Image 2.1 (`790c926`), this repo | 18.14 | 14.01 | 41.68 | 80.48 | 88.25 |
| | Qwen-Image-Edit, RINO Table 3 | 21.99 | 18.14 | – | – | – |
| | StableNormal, RINO Table 3 | 13.70 | 9.46 | 63.45 | 86.31 | 92.11 |

StableNormal threshold columns are from the StableNormal paper (Table 2). No generation or scoring failures occurred.

## Fine-tuning data

Union multi-task training data for instruction-tuning the editor: segmentation and depth (NYUv2, Virtual KITTI 2),
surface normals (Hypersim) and human pose estimation (COCO, RINO protocol) as (image, instruction, target image)
triplets whose targets and prompts match the evaluation configs above. Build it with
`bash scripts/build_finetune_data.sh` (about 12 GB, verified against checksums) and load it with
`finetune_data.dataset.FinetuneDataset`. It also adds in-domain test configs (`configs/indomain_*.yaml`), DIODE depth
(`configs/diode_depth.yaml`) and RINO-style pose estimation (`configs/coco_pose_estimation.yaml`).
Full guide, options and known issues: [`finetune_data/README.md`](finetune_data/README.md).

## Tests

```bash
uv run python -m pytest -q
```

The test suite covers plugin contracts, transforms, mocked provider requests, Qwen configuration and mocked generation, archive handling, and run lifecycle. Real provider and CUDA calls are deliberately excluded from automated tests.
