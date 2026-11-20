#!/usr/bin/env bash
# Download the Virtual KITTI 2 subset used for fine-tuning into raw/vkitti2/ (no login).
# Each archive (7.5 + 8.1 + 1.0 GB) is downloaded with resume support (the server drops long
# connections), then only the needed members are extracted and the archive is deleted:
#   - Camera_0 only (Camera_1 is the other half of the stereo pair)
#   - variations clone, fog, morning, overcast, rain, sunset (not the rotated cameras)
#   - frames whose number ends in 0 or 5 (every 5th frame; consecutive frames are near-identical)
# About 1 GB is kept; up to 8.1 GB is needed temporarily. Set KEEP_ARCHIVES=1 to keep the archives,
# VARIATIONS="clone fog" or FRAME_GLOB="?????" (every frame) to change the subset.
# Source: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/vkitti2}"
BASE=https://download.europe.naverlabs.com/virtual_kitti_2.0.3
VARIATIONS="${VARIATIONS:-clone fog morning overcast rain sunset}"
FRAME_GLOB="${FRAME_GLOB:-????[05]}"
ARCHIVES="$TARGET/.archives"
mkdir -p "$TARGET" "$ARCHIVES"

wildcards=()
if tar --version 2>/dev/null | grep -q GNU; then wildcards=(--wildcards); fi

download() {  # resumable; retries until the file has the server's full size
  local archive=$1 url="$BASE/$1" dest="$ARCHIVES/$1" expected attempt
  expected=$(curl -fsSIL "$url" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r' | tail -1)
  for attempt in $(seq 1 20); do
    [ -f "$dest" ] && [ "$(wc -c < "$dest" | tr -d ' ')" = "$expected" ] && return 0
    echo "Downloading $archive (attempt $attempt) ..."
    curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url" || sleep 5
  done
  echo "Failed to download $archive" >&2
  return 1
}

fetch() {  # archive, modality folder, file prefix, extension
  local archive=$1 modality=$2 prefix=$3 ext=$4 patterns=()
  for v in $VARIATIONS; do
    patterns+=("Scene*/$v/frames/$modality/Camera_0/${prefix}_${FRAME_GLOB}.$ext")
  done
  download "$archive"
  echo "Extracting $modality ..."
  tar -x -f "$ARCHIVES/$archive" -C "$TARGET" ${wildcards[@]+"${wildcards[@]}"} "${patterns[@]}"
  [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$ARCHIVES/$archive"
}

download vkitti_2.0.3_textgt.tar.gz
tar -xz -f "$ARCHIVES/vkitti_2.0.3_textgt.tar.gz" -C "$TARGET"
fetch vkitti_2.0.3_classSegmentation.tar classSegmentation classgt png
fetch vkitti_2.0.3_depth.tar depth depth png
fetch vkitti_2.0.3_rgb.tar rgb rgb jpg

for modality in rgb depth classSegmentation; do
  echo "$modality: $(find "$TARGET" -path "*/frames/$modality/*" -type f | wc -l | tr -d ' ') files"
done
echo "Next: uv run python -m finetune_data.sources.vkitti2 $TARGET data/finetune_canonical"
