"""NExT-OE test (lmms-lab/NExTQA parquet) joined with the NExT-QA MC test on (video, qid)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vjaudit.schemas import Item

from ._common import check_unique, make_item


def load_nextoe(raw_dir: Path, split: str = "test") -> list[Item]:
    raw_dir = Path(raw_dir)
    oe = pd.read_parquet(raw_dir / "OE" / f"{split}-00000-of-00001.parquet")
    mc_path = raw_dir / "MC" / f"{split}-00000-of-00001.parquet"
    mc = {}
    if mc_path.exists():
        for r in pd.read_parquet(mc_path).to_dict("records"):
            mc[(str(r["video"]), int(r["qid"]))] = r
    items = []
    for r in oe.to_dict("records"):
        video, qid = str(r["video"]), int(r["qid"])
        refs = [str(r["answer"])]
        extra = r.get("additional_ref_answer")
        if isinstance(extra, str) and extra.strip():
            refs.append(extra)
        m = mc.get((video, qid))
        items.append(make_item(
            "nextoe", split, f"{video}_{qid}", video_id=video, question=r["question"], answers_ref=refs,
            qtype=str(r["type"]),
            mc_options=[str(m[f"a{k}"]) for k in range(5)] if m else None,
            mc_answer_idx=int(m["answer"]) if m else None,
        ))
    return check_unique(items)
