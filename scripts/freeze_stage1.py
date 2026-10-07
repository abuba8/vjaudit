"""Freeze Stage-1 artifacts (plan Task 14). Usage: uv run python scripts/freeze_stage1.py [out_dir]

Writes sorted item/generation/judgecall exports and api_cache.tar.gz to out_dir (default
artifacts/freeze-stage1), and artifacts/manifest.jsonl (tracked in git). Then copy out_dir to two
storage locations and record paths + checksums in docs/stage1_freeze.md.
"""
import sys
from pathlib import Path

from vjaudit.freeze import freeze
from vjaudit.profile import load_profile

if __name__ == "__main__":
    prof = load_profile()  # $VJAUDIT_PROFILE, default server
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else prof.artifacts / "freeze-stage1"
    manifest = Path("artifacts/manifest.jsonl") if prof.name == "server" else Path(f"artifacts/manifest_{prof.name}.jsonl")
    for e in freeze(prof.store(), prof.api_cache, out, manifest=manifest):
        print(f"{e.kind:12s} {e.n_records:8d}  {e.sha256}  {e.path}")
