"""R-SP0-01: are the prior pipeline's generations byte-identical across prompting conditions?"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .load import LegacyCell

COLUMNS = ["cond_a", "set_a", "cond_b", "set_b", "dataset", "frames", "n", "pct_identical"]


def pairwise_identity(cells: list[LegacyCell], split: str) -> pd.DataFrame:
    """Percent of shared indices whose "Generated Answer" is byte-identical, per cell pair."""
    rows = []
    groups: dict[tuple[str, int], list[LegacyCell]] = {}
    for c in cells:
        if c.split == split:
            groups.setdefault((c.dataset, c.frames), []).append(c)
    for (dataset, frames), group in sorted(groups.items()):
        loaded = [(c, c.gen()) for c in group]
        for (a, ga), (b, gb) in combinations(loaded, 2):
            shared = sorted(set(ga) & set(gb))
            same = sum(ga[k]["Generated Answer"] == gb[k]["Generated Answer"] for k in shared)
            rows.append([a.condition, a.set, b.condition, b.set, dataset, frames, len(shared),
                         round(100.0 * same / len(shared), 2) if shared else float("nan")])
    return pd.DataFrame(rows, columns=COLUMNS)
