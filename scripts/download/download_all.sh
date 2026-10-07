#!/usr/bin/env bash
# Download all three Stage-1 benchmarks and create configs/local/paths.yaml if it does not exist.
set -euo pipefail
source "$(dirname "$0")/common.sh"
here="$(dirname "$0")"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--annotations-only] [--help]
Runs download_msvd.sh, download_nextqa.sh and download_anetqa.sh (ActivityNet videos last: ~130 GB),
then copies configs/paths.example.yaml to configs/local/paths.yaml if missing.
Set DATA_ROOT to download somewhere other than ./data (then edit configs/local/paths.yaml to match).
USAGE
}
flag=""
for a in "$@"; do
  case "$a" in
    --help|-h) usage; exit 0 ;;
    --annotations-only) flag="--annotations-only" ;;
    *) usage; exit 2 ;;
  esac
done

mkdir -p "$REPO_ROOT/configs/local"
if [ ! -f "$REPO_ROOT/configs/local/paths.yaml" ]; then
  cp "$REPO_ROOT/configs/paths.example.yaml" "$REPO_ROOT/configs/local/paths.yaml"
  log "created configs/local/paths.yaml from configs/paths.example.yaml"
fi
bash "$here/download_msvd.sh" $flag
bash "$here/download_nextqa.sh" $flag
bash "$here/download_anetqa.sh" $flag
log "all downloads finished. Next: make data-check (add PROFILE=mac on a Mac)"
