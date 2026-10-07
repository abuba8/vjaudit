#!/usr/bin/env bash
# Shared helpers for the dataset download scripts. Sourced, not executed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(basename "$0")" "$*"; }
warn() { printf '\033[1;33m[%s] WARNING:\033[0m %s\n' "$(basename "$0")" "$*" >&2; }
die()  { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "$(basename "$0")" "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required: $2"; }

# Run the project's Python: the active conda environment's python (any env except base), or plain python when
# uv is not installed; otherwise `uv run python`. Matches the Makefile's RUN rule.
run_python() {
  if { [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != base ]; } || ! command -v uv >/dev/null 2>&1; then
    python "$@"
  else
    uv run python "$@"
  fi
}

# Hugging Face dataset download: uses `hf`, `huggingface-cli`, or `uvx --from huggingface_hub hf`.
hf_download() {  # hf_download <repo> <local_dir> [extra args...]
  local repo=$1 dir=$2; shift 2
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --repo-type dataset --local-dir "$dir" "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --repo-type dataset --local-dir "$dir" "$@"
  elif command -v uvx >/dev/null 2>&1; then
    uvx --from huggingface_hub hf download "$repo" --repo-type dataset --local-dir "$dir" "$@"
  else
    die "need the Hugging Face CLI: 'uv tool install huggingface_hub' (or pip install huggingface_hub)"
  fi
}

# MEGA download: megatools >= 1.11 (`megatools dl`) or older `megadl`.
mega_download() {  # mega_download <url> <dir>
  local url=$1 dir=$2
  mkdir -p "$dir"
  if command -v megatools >/dev/null 2>&1; then
    megatools dl --path "$dir" "$url"
  elif command -v megadl >/dev/null 2>&1; then
    megadl --path "$dir" "$url"
  else
    die "need megatools to fetch $url (Ubuntu: sudo apt install megatools; macOS: brew install megatools)"
  fi
}

# Unzip every archive in <src_dir> matching <glob> into <dst_dir>; skip if <dst_dir> already has files.
unzip_all() {  # unzip_all <src_dir> <glob> <dst_dir>
  local src=$1 pat=$2 dst=$3
  need unzip "sudo apt install unzip"
  mkdir -p "$dst"
  shopt -s nullglob
  for z in "$src"/$pat; do
    log "unzipping $(basename "$z") -> $dst"
    unzip -q -n "$z" -d "$dst"
  done
  shopt -u nullglob
}
