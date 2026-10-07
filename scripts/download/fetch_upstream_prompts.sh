#!/usr/bin/env bash
# Fetch the upstream Video-ChatGPT evaluation scripts and verify our T-VCG prompt against them
# (configs/prompts/judge/t_vcg_*.txt). Marks t-vcg as verified in source.yaml when they match.
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--help]
Clones mbzuai-oryx/Video-ChatGPT (depth 1) into \$DATA_ROOT/upstream and runs
  python -m vjaudit.cli judge verify-vcg --upstream <activitynet script> --upstream <msvd script> ...
(python from your active conda environment, or via uv when no conda environment is active)
USAGE
}
[ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] && { usage; exit 0; }

need git "install git"
dst="$DATA_ROOT/upstream/Video-ChatGPT"
rm -rf "$dst"
git clone --depth 1 https://github.com/mbzuai-oryx/Video-ChatGPT.git "$dst"
sha=$(git -C "$dst" rev-parse HEAD)
qe="$dst/quantitative_evaluation"
args=(--upstream "$qe/evaluate_activitynet_qa.py")
for f in "$qe"/evaluate_*qa*.py; do
  [ "$f" = "$qe/evaluate_activitynet_qa.py" ] || args+=(--upstream "$f")
done
cd "$REPO_ROOT"
run_python -m vjaudit.cli judge verify-vcg "${args[@]}" --commit "$sha"
