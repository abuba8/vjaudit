"""Usage: uv run python scripts/sp0_report.py [configs/local/legacy.yaml] [docs/sp0_report.md]"""
import sys
from pathlib import Path

from vjaudit.legacy.report import write_sp0_report

if __name__ == "__main__":
    cfg = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/local/legacy.yaml")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/sp0_report.md")
    write_sp0_report(cfg, Path("tests/legacy/expected_tables.json"), out)
    print(f"wrote {out}")
