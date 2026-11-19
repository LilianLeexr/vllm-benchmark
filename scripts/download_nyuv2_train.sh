#!/usr/bin/env bash
# Download the official NYU Depth V2 labeled set (1,449 images, ~2.8 GB) plus the 40-class
# labels and the official train/test split into raw/nyuv2_train/, for fine-tuning data.
# The 654 test images are the Eigen test split our depth/normals evaluators use; the
# fine-tuning store marks them split="test" so they never enter training.
# Sources: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/ and
#          https://github.com/ankurhanda/nyuv2-meta-data (labels40.mat, splits.mat)
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/nyuv2_train}"
META=https://github.com/ankurhanda/nyuv2-meta-data/raw/master
mkdir -p "$TARGET"
for f in labels40.mat classMapping40.mat splits.mat; do
  curl -fsSL --retry 5 -o "$TARGET/$f" "$META/$f"
done
curl -fL --retry 5 -C - -o "$TARGET/nyu_depth_v2_labeled.mat" \
  http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

size=$(wc -c < "$TARGET/nyu_depth_v2_labeled.mat" | tr -d ' ')
[ "$size" -eq 2972037809 ] || { echo "Unexpected size for nyu_depth_v2_labeled.mat: $size" >&2; exit 1; }
echo "NYUv2 labeled set ready in $TARGET. Next:"
echo "  uv run python -m finetune_data.sources.nyuv2 $TARGET data/finetune_canonical"
