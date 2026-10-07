"""Pin HF revision SHAs in a profile's generators.yaml / judges.yaml (R-EXP-01, R-GLB-03).

Only entries that have a ``revision:`` line are pinned; API judges have none and stay untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def hf_resolver(repo: str) -> str:
    from huggingface_hub import model_info  # gpu extra

    return model_info(repo).sha


def pin_revisions(path: Path, resolver: Callable[[str], str] = hf_resolver) -> dict[str, str]:
    """Rewrite each ``revision:`` line with resolver(repo), where repo is the entry's ``hf_repo:`` (generators)
    or ``model:`` (judges); comments and layout are kept."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    pinned: dict[str, str] = {}
    cur_id = cur_repo = None
    for i, line in enumerate(lines):
        if m := re.match(r"\s*-\s*id:\s*(\S+)", line):
            cur_id, cur_repo = m.group(1), None
        elif m := re.match(r"\s*(?:hf_repo|model):\s*(\S+)", line):
            cur_repo = m.group(1)
        elif m := re.match(r"(\s*revision:\s*)\S+", line):
            if cur_repo is None:
                raise ValueError(f"line {i + 1}: revision before hf_repo for {cur_id}")
            sha = resolver(cur_repo)
            lines[i] = m.group(1) + sha
            pinned[cur_id] = sha
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return pinned
