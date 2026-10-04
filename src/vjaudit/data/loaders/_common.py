from __future__ import annotations

from vjaudit.schemas import Item
from vjaudit.store import stamp

LOADER_VERSION = 1


def make_item(benchmark: str, split: str, native_qid: str, **fields) -> Item:
    """Build a stamped Item with id "{benchmark}:{split}:{native_qid}"; video_path is resolved later."""
    rec = dict(item_id=f"{benchmark}:{split}:{native_qid}", benchmark=benchmark, split=split,
               video_path="", **fields)
    return Item(**stamp(rec, {"loader": benchmark, "version": LOADER_VERSION}))


def check_unique(items: list[Item]) -> list[Item]:
    seen: set[str] = set()
    for it in items:
        if it.item_id in seen:
            raise ValueError(f"duplicate native qid: {it.item_id}")
        seen.add(it.item_id)
    return items
