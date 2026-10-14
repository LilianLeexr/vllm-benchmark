#!/usr/bin/env bash
# Download the ADE20K scene-parsing benchmark (MIT, ~920 MB, no login) into raw/ade20k/.
# Source: http://sceneparsing.csail.mit.edu/ (ADEChallengeData2016)
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/ade20k}"
URL=http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
PARENT="$(dirname "$TARGET")"

mkdir -p "$PARENT"
curl -fL --retry 5 -C - -o "$PARENT/ADEChallengeData2016.zip" "$URL"
unzip -oq "$PARENT/ADEChallengeData2016.zip" -d "$PARENT"
rm -rf "$TARGET"
mv "$PARENT/ADEChallengeData2016" "$TARGET"
rm "$PARENT/ADEChallengeData2016.zip"

images=$(find "$TARGET/images/validation" -name '*.jpg' | wc -l)
labels=$(find "$TARGET/annotations/validation" -name '*.png' | wc -l)
echo "ADE20K validation: $images images, $labels annotations in $TARGET"
[ "$images" -eq 2000 ] && [ "$labels" -eq 2000 ] || { echo "Expected 2000 images and annotations" >&2; exit 1; }
