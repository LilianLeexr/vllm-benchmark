#!/usr/bin/env bash
# Download the NYUv2 Eigen test split (654 images) packaged by Marigold into raw/nyuv2/.
# Source: https://github.com/prs-eth/Marigold#-evaluation-on-test-datasets
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/nyuv2}"
BASE=https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/nyuv2
SPLIT=https://raw.githubusercontent.com/prs-eth/Marigold/main/data_split/nyu_depth/labeled/filename_list_test.txt

mkdir -p "$TARGET"
curl -fL --retry 5 -C - -o "$TARGET/nyu_labeled_extracted.tar" "$BASE/nyu_labeled_extracted.tar"
curl -fL --retry 5 -o "$TARGET/filename_list_test.txt" "$SPLIT"
tar -xf "$TARGET/nyu_labeled_extracted.tar" -C "$TARGET" --wildcards '*test/*'
rm "$TARGET/nyu_labeled_extracted.tar"

count=$(grep -c . "$TARGET/filename_list_test.txt")
found=$(find "$TARGET/test" -name 'rgb_*.png' | wc -l)
echo "NYUv2 test split: $found images found, $count listed in $TARGET/filename_list_test.txt"
[ "$found" -ge "$count" ] || { echo "Missing NYUv2 images" >&2; exit 1; }
