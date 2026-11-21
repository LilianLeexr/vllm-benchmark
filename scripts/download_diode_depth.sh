#!/usr/bin/env bash
# Download the DIODE val depth set packaged by Marigold (771 images: 325 indoors + 446 outdoor) into
# raw/diode_depth/, with Marigold's split lists. Same images, depth and validity masks as Marigold's /
# Depth Anything's DIODE numbers. The 6.4 GB tar is downloaded with resume support, extracted, deleted.
# Source: https://github.com/prs-eth/Marigold (evaluation datasets)
# Usage: scripts/download_diode_depth.sh [target_dir]     KEEP_ARCHIVES=1 keeps the tar
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/diode_depth}"
TAR_URL="https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/diode/diode_val.tar"
SPLITS="https://raw.githubusercontent.com/prs-eth/Marigold/2bfbdeae5d10a50b71f1ba20c865d46e480f6010/data_split/diode_depth"
TAR="$TARGET/.archives/diode_val.tar"
mkdir -p "$TARGET/.archives"

for split in all indoor outdoor; do
  curl -fsSL "$SPLITS/diode_val_${split}_filename_list.txt" -o "$TARGET/diode_val_${split}_filename_list.txt"
done
if [ "$(find "$TARGET" -name '*_depth.npy' | wc -l | tr -d ' ')" -ge 771 ]; then
  echo "DIODE depth already present in $TARGET"; exit 0
fi
expected=$(curl -fsSIL "$TAR_URL" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r' | tail -1)
for attempt in $(seq 1 20); do
  [ -f "$TAR" ] && [ "$(wc -c < "$TAR" | tr -d ' ')" = "$expected" ] && break
  echo "Downloading diode_val.tar (attempt $attempt) ..."
  curl -fL --retry 5 --retry-delay 5 -C - -o "$TAR" "$TAR_URL" || sleep 5
done
[ "$(wc -c < "$TAR" | tr -d ' ')" = "$expected" ] || { echo "download failed" >&2; exit 1; }
tar -x -f "$TAR" -C "$TARGET"
[ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$TAR"
echo "DIODE depth: $(find "$TARGET" -name '*_depth.npy' | wc -l | tr -d ' ') depth maps in $TARGET"
