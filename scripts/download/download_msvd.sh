#!/usr/bin/env bash
# MSVD-QA (Xu et al., ACM MM 2017): official QA files (MEGA, from github.com/xudejing/video-question-answering)
# and the MSVD YouTube clips (official tarball from the UT Austin MSVD page).
#
# Output:
#   data/msvd/annotations/test_qa.json, train_qa.json, val_qa.json, youtube_mapping.txt
#   data/msvd/videos/YouTubeClips/*.avi        (~1.7 GB; 1,970 clips)
set -euo pipefail
source "$(dirname "$0")/common.sh"

QA_URL='https://mega.nz/#!QmxFwBTK!Cs7cByu_Qo42XJOsv0DjiEDMiEm8m69h60caDYnT_PQ'
MAPPING_URL='https://mega.nz/#!QrowUADZ!oFfW_M5wAFsfuFDEJAIa2BeFVHYO0vxit3CMkHFOSfw'
VIDEOS_URL='https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar'
ANN_DIR="$DATA_ROOT/msvd/annotations"     # = msvd_raw in configs/paths.example.yaml (data/msvd/annotations)
VID_DIR="$DATA_ROOT/msvd/videos"          # = msvd_videos (data/msvd/videos)

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--annotations-only | --videos-only] [--help]
Downloads MSVD-QA annotations (needs megatools) and the YouTubeClips.tar videos (needs wget or curl)
into \$DATA_ROOT/msvd (default: data/msvd). Safe to re-run: finished steps are skipped.
USAGE
}

ann=1; vid=1
for a in "$@"; do
  case "$a" in
    --help|-h) usage; exit 0 ;;
    --annotations-only) vid=0 ;;
    --videos-only) ann=0 ;;
    *) usage; exit 2 ;;
  esac
done

if [ "$ann" = 1 ]; then
  if [ -f "$ANN_DIR/test_qa.json" ] && [ -f "$ANN_DIR/youtube_mapping.txt" ]; then
    log "annotations present in $ANN_DIR"
  else
    tmp="$ANN_DIR/_mega"; mkdir -p "$tmp"
    log "downloading MSVD-QA QA files from MEGA"
    mega_download "$QA_URL" "$tmp"
    mega_download "$MAPPING_URL" "$tmp"
    shopt -s nullglob
    for z in "$tmp"/*.zip; do need unzip "sudo apt install unzip"; unzip -q -o "$z" -d "$tmp"; done
    shopt -u nullglob
    for f in test_qa.json train_qa.json val_qa.json youtube_mapping.txt; do
      found=$(find "$tmp" -name "$f" -print -quit)
      [ -n "$found" ] && cp "$found" "$ANN_DIR/$f" || warn "$f not found in the MEGA download"
    done
    [ -f "$ANN_DIR/test_qa.json" ] || die "test_qa.json missing after download; see $tmp"
  fi
fi

if [ "$vid" = 1 ]; then
  if [ -d "$VID_DIR/YouTubeClips" ] && [ -n "$(ls -A "$VID_DIR/YouTubeClips" 2>/dev/null)" ]; then
    log "videos present in $VID_DIR/YouTubeClips"
  else
    mkdir -p "$VID_DIR"
    tarball="$VID_DIR/YouTubeClips.tar"
    if [ ! -f "$tarball" ]; then
      log "downloading $VIDEOS_URL (~1.7 GB)"
      if command -v wget >/dev/null 2>&1; then wget -c -O "$tarball" "$VIDEOS_URL"
      else need curl "install curl or wget"; curl -L -C - -o "$tarball" "$VIDEOS_URL"; fi
    fi
    log "extracting $tarball"
    tar -xf "$tarball" -C "$VID_DIR"
    log "$(find "$VID_DIR" -name '*.avi' | wc -l) .avi clips in $VID_DIR"
  fi
fi
log "done. MSVD-QA: annotations=$ANN_DIR videos=$VID_DIR"
