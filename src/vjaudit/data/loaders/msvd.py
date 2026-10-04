"""MSVD-QA (Xu et al. 2017) official release: test_qa.json + youtube_mapping.txt."""
from __future__ import annotations

import json
import re
from pathlib import Path

from vjaudit.schemas import Item

from ._common import check_unique, make_item

QTYPES = ("what", "who", "how", "when", "where")


def qtype_msvd(question: str) -> str:
    m = re.match(r"\s*([a-z]+)", question.lower())
    return m.group(1) if m and m.group(1) in QTYPES else "other"


def load_msvd(raw_dir: Path, split: str = "test") -> list[Item]:
    raw_dir = Path(raw_dir)
    mapping = {}
    for line in (raw_dir / "youtube_mapping.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            clip, vid = line.split()
            mapping[vid] = clip  # "vid12" -> "<youtube_clip>"
    rows = json.loads((raw_dir / f"{split}_qa.json").read_text(encoding="utf-8"))
    items = [
        make_item("msvd", split, str(r["id"]), video_id=mapping[f"vid{r['video_id']}"],
                  question=r["question"], answers_ref=[r["answer"]], qtype=qtype_msvd(r["question"]))
        for r in rows
    ]
    return check_unique(items)
