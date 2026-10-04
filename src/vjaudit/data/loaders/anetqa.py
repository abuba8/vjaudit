"""ActivityNet-QA (Yu et al. 2019) official release: test_q.json + test_a.json joined on question_id."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from vjaudit.schemas import Item

from ._common import check_unique, make_item

log = logging.getLogger(__name__)
TYPE_NAMES = ["motion", "spatial", "temporal", "yes_no", "color", "object", "location", "number", "other"]


def _qtype(t) -> str:
    try:
        return TYPE_NAMES[int(t)]
    except (ValueError, IndexError, TypeError):
        return str(t)


def load_anetqa(raw_dir: Path, split: str = "test") -> list[Item]:
    raw_dir = Path(raw_dir)
    qs = json.loads((raw_dir / f"{split}_q.json").read_text(encoding="utf-8"))
    ans = {a["question_id"]: a for a in json.loads((raw_dir / f"{split}_a.json").read_text(encoding="utf-8"))}
    items, dropped = [], []
    for q in qs:
        a = ans.get(q["question_id"])
        if a is None:
            dropped.append(q["question_id"])
            continue
        items.append(make_item("anetqa", split, q["question_id"], video_id=f"v_{q['video_name']}",
                               question=q["question"], answers_ref=[a["answer"]], qtype=_qtype(a["type"])))
    if dropped:
        log.warning("anetqa: %d questions without answers dropped: %s", len(dropped), ", ".join(dropped[:20]))
    return check_unique(items)
