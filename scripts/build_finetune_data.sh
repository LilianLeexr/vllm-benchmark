#!/usr/bin/env bash
# Build the whole fine-tuning store (about 12 GB) from the public sources and check it against our
# checksums, so every machine gets pixel-identical data. Takes about 2 hours, mostly downloads
# (NYUv2 2.8 GB, VKITTI2 ~17 GB streamed, Hypersim ~17 GB streamed, COCO ~0.7 GB). Re-running resumes:
# sources already built are skipped (FORCE=1 rebuilds), and the Hypersim / COCO steps skip done frames.
#
#   bash scripts/build_finetune_data.sh                  # -> data/finetune_canonical
#   ROOT=/big/disk/finetune_canonical bash scripts/build_finetune_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${ROOT:-data/finetune_canonical}"
PY="uv run python"

built() { [ "${FORCE:-0}" != 1 ] && [ -f "$ROOT/$1/meta.json" ] && [ -f "$ROOT/$1/manifest.jsonl" ]; }

if built nyuv2; then echo "nyuv2: already built"; else
  bash scripts/download_nyuv2_train.sh
  $PY -m finetune_data.sources.nyuv2 raw/nyuv2_train "$ROOT"
fi
if built vkitti2; then echo "vkitti2: already built"; else
  bash scripts/download_vkitti2.sh
  $PY -m finetune_data.sources.vkitti2 raw/vkitti2 "$ROOT"
fi
if built hypersim; then echo "hypersim: already built"; else
  $PY -m finetune_data.sources.hypersim "$ROOT" --train-frames 4000
fi
if built coco_pose; then echo "coco_pose: already built"; else
  $PY -m finetune_data.sources.coco_pose raw/coco2017 "$ROOT"
fi
$PY -m finetune_data.verify "$ROOT"
