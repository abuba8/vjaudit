"""Machine-specific locations (configs/local/paths.yaml, gitignored)."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATHS_FILE = Path("configs/local/paths.yaml")


def load_paths(path: Path | str = DEFAULT_PATHS_FILE) -> dict[str, Path]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: copy configs/paths.example.yaml to {DEFAULT_PATHS_FILE} and edit it")
    return {k: Path(v) for k, v in yaml.safe_load(p.read_text(encoding="utf-8")).items()}
