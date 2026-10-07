#!/usr/bin/env bash
# NExT-QA (Xiao et al., CVPR 2021): open-ended (OE) and multiple-choice (MC) test annotations and the
# videos, from the lmms-lab/NExTQA Hugging Face mirror (OE/ and MC/ parquet + videos.zip, ~6.5 GB).
#
# Output:
#   data/nextqa/annotations/OE/test-00000-of-00001.parquet, MC/test-00000-of-00001.parquet
#   data/nextqa/videos/...                    (extracted from videos.zip)
set -euo pipefail
source "$(dirname "$0")/common.sh"

ANN_DIR="$DATA_ROOT/nextqa/annotations"   # = nextqa_raw (data/nextqa/annotations)
VID_DIR="$DATA_ROOT/nextqa/videos"        # = nextqa_videos (data/nextqa/videos)

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--annotations-only] [--keep-zips] [--help]
Downloads lmms-lab/NExTQA (needs the Hugging Face CLI and unzip) into \$DATA_ROOT/nextqa.
USAGE
}

vid=1; keep=0
for a in "$@"; do
  case "$a" in
    --help|-h) usage; exit 0 ;;
    --annotations-only) vid=0 ;;
    --keep-zips) keep=1 ;;
    *) usage; exit 2 ;;
  esac
done

mkdir -p "$ANN_DIR"
log "downloading lmms-lab/NExTQA annotations (OE + MC parquet)"
hf_download lmms-lab/NExTQA "$ANN_DIR" --include "OE/*" --include "MC/*" --include "README.md"
[ -f "$ANN_DIR/OE/test-00000-of-00001.parquet" ] || die "OE test parquet missing in $ANN_DIR/OE"

if [ "$vid" = 1 ]; then
  if [ -d "$VID_DIR" ] && [ -n "$(ls -A "$VID_DIR" 2>/dev/null)" ]; then
    log "videos present in $VID_DIR"
  else
    log "downloading videos.zip (~6.5 GB)"
    hf_download lmms-lab/NExTQA "$ANN_DIR" --include "videos.zip"
    unzip_all "$ANN_DIR" "videos.zip" "$VID_DIR"
    [ "$keep" = 1 ] || rm -f "$ANN_DIR/videos.zip"
  fi
  log "$(find "$VID_DIR" -type f -name '*.mp4' | wc -l) .mp4 files in $VID_DIR"
fi
log "done. NExT-QA: annotations=$ANN_DIR videos=$VID_DIR"
