#!/usr/bin/env bash
# Download Cityscapes fine annotations and images (~11 GB) into raw/cityscapes/.
# Requires a free account at https://www.cityscapes-dataset.com/register/; the official
# csDownload tool asks for the username and password on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/cityscapes}"
mkdir -p "$TARGET"
uvx --from cityscapesscripts csDownload -d "$TARGET" gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
for archive in gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip; do
  unzip -oq "$TARGET/$archive" -d "$TARGET"
done

images=$(find "$TARGET/leftImg8bit/val" -name '*_leftImg8bit.png' | wc -l)
labels=$(find "$TARGET/gtFine/val" -name '*_gtFine_labelIds.png' | wc -l)
echo "Cityscapes val: $images images, $labels label maps in $TARGET"
[ "$images" -eq 500 ] && [ "$labels" -eq 500 ] || { echo "Expected 500 images and label maps" >&2; exit 1; }
rm "$TARGET/gtFine_trainvaltest.zip" "$TARGET/leftImg8bit_trainvaltest.zip"
