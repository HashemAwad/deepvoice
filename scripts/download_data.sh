#!/usr/bin/env bash
# Download ASVspoof 2019 LA (7.6 GB) and In-the-Wild (8.2 GB) into ./data (gitignored).
# Resumable (wget -c): rerun after an interrupted download. Needs wget and unzip.
#   ./scripts/download_data.sh              delete the zips after extracting
#   ./scripts/download_data.sh --keep-zips  keep them
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP=0
[ "${1:-}" = "--keep-zips" ] && KEEP=1
for tool in wget unzip; do
  command -v "$tool" >/dev/null || { echo "missing '$tool'; install it first" >&2; exit 1; }
done

LA_URL="https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
LA_BYTES=7640952520
ITW_URL="https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
ITW_BYTES=8161489023

mkdir -p data
cd data

# fetch <url> <zip> <expected_bytes> <extracted_dir>
fetch() {
  if [ -d "$4" ]; then echo "$4 already extracted, skipping"; return; fi
  wget -c -q --show-progress -O "$2" "$1"
  got=$(wc -c < "$2" | tr -d ' ')
  if [ "$got" != "$3" ]; then
    echo "$2 is $got bytes, expected $3; rerun to resume" >&2
    exit 1
  fi
  unzip -q "$2"
  [ "$KEEP" = 1 ] || rm -f "$2"
}

fetch "$LA_URL"  LA.zip "$LA_BYTES" LA
fetch "$ITW_URL" release_in_the_wild.zip "$ITW_BYTES" release_in_the_wild

echo "flac in LA:  $(find LA -name '*.flac' | wc -l)  (expect 122299)"
echo "wav in ITW:  $(find release_in_the_wild -name '*.wav' | wc -l)  (expect 31779)"
