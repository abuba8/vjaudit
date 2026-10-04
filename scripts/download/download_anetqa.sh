#!/usr/bin/env bash
# ActivityNet-QA (Yu et al., AAAI 2019): official QA files from github.com/MILVLG/activitynet-qa and the
# 800 test videos from the lmms-lab/ActivityNetQA Hugging Face mirror (28 zip chunks, ~130 GB).
#
# Output:
#   data/anetqa/annotations/{train,val,test}_{q,a}.json
#   data/anetqa/videos/...                      (test videos, extracted from the zip chunks)
set -euo pipefail
source "$(dirname "$0")/common.sh"

ANN_DIR="$DATA_ROOT/anetqa/annotations"   # = anetqa_raw (data/anetqa/annotations)
VID_DIR="$DATA_ROOT/anetqa/videos"        # = anetqa_videos (data/anetqa/videos)
HF_DIR="$DATA_ROOT/anetqa/hf"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--annotations-only | --videos-only] [--keep-zips] [--help]
Annotations: git clone of MILVLG/activitynet-qa (needs git).
Videos: Hugging Face dataset lmms-lab/ActivityNetQA, 28 zips, ~130 GB download + ~130 GB extracted
(needs the Hugging Face CLI and unzip). Zips are deleted after extraction unless --keep-zips.
USAGE
}

ann=1; vid=1; keep=0
for a in "$@"; do
  case "$a" in
    --help|-h) usage; exit 0 ;;
    --annotations-only) vid=0 ;;
    --videos-only) ann=0 ;;
    --keep-zips) keep=1 ;;
    *) usage; exit 2 ;;
  esac
done

if [ "$ann" = 1 ]; then
  if [ -f "$ANN_DIR/test_q.json" ] && [ -f "$ANN_DIR/test_a.json" ]; then
    log "annotations present in $ANN_DIR"
  else
    need git "install git"
    tmp="$DATA_ROOT/anetqa/_repo"
    rm -rf "$tmp"
    git clone --depth 1 https://github.com/MILVLG/activitynet-qa.git "$tmp"
    mkdir -p "$ANN_DIR"
    cp "$tmp"/dataset/*.json "$ANN_DIR"/
    git -C "$tmp" rev-parse HEAD > "$ANN_DIR/SOURCE_COMMIT"
    log "annotations from MILVLG/activitynet-qa @ $(cat "$ANN_DIR/SOURCE_COMMIT")"
  fi
fi

if [ "$vid" = 1 ]; then
  avail=$(df -Pk "$DATA_ROOT" 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || echo 0)
  [ "${avail:-0}" -lt 270 ] && warn "only ${avail} GB free under $DATA_ROOT; the videos need ~260 GB during extraction"
  mkdir -p "$HF_DIR" "$VID_DIR"
  log "downloading lmms-lab/ActivityNetQA video chunks (resumable)"
  hf_download lmms-lab/ActivityNetQA "$HF_DIR" --include "*.zip"
  unzip_all "$HF_DIR" "*.zip" "$VID_DIR"
  [ "$keep" = 1 ] || find "$HF_DIR" -name '*.zip' -delete
  log "$(find "$VID_DIR" -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' \) | wc -l) video files in $VID_DIR"
fi
log "done. ActivityNet-QA: annotations=$ANN_DIR videos=$VID_DIR"
