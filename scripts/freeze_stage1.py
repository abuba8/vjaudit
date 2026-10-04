"""Freeze Stage-1 artifacts (plan Task 14). Usage: uv run python scripts/freeze_stage1.py [out_dir]

Writes sorted item/generation/judgecall exports and api_cache.tar.gz to out_dir (default
artifacts/freeze-stage1), and artifacts/manifest.jsonl (tracked in git). Then copy out_dir to two
storage locations and record paths + checksums in docs/stage1_freeze.md.
"""
import sys
from pathlib import Path

from vjaudit.freeze import freeze
from vjaudit.paths import load_paths
from vjaudit.store import ArtifactStore

if __name__ == "__main__":
    p = load_paths()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else p["artifacts"] / "freeze-stage1"
    for e in freeze(ArtifactStore(p["artifacts"]), p["api_cache"], out, manifest=Path("artifacts/manifest.jsonl")):
        print(f"{e.kind:12s} {e.n_records:8d}  {e.sha256}  {e.path}")
