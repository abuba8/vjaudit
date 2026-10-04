"""Nested stratified samples (main ⊃ annotated ⊃ vid) and the dev/test split (R-ANN-05, R-ANN-16)."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from vjaudit.schemas import Item


def allocate(counts: dict[str, int], n: int, min_per: int = 0) -> dict[str, int]:
    """Split ``n`` across strata: largest-remainder proportional, then raise strata below
    ``min_per`` (capped at their size) one unit at a time from the largest current allocation."""
    total = sum(counts.values())
    n = min(n, total)
    if n == 0:
        return {k: 0 for k in counts}
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: math.floor(q) for k, q in quotas.items()}
    for k in sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    for k in sorted(counts):
        target = min(min_per, counts[k])
        while alloc[k] < target:
            donor = max((d for d in alloc if d != k and alloc[d] > min(min_per, counts[d])),
                        key=lambda d: (alloc[d], d), default=None)
            if donor is None:
                break
            alloc[donor] -= 1
            alloc[k] += 1
    return alloc


def _strata(items: list[Item]) -> dict[str, list[Item]]:
    s: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        s[it.qtype].append(it)
    return {k: sorted(v, key=lambda i: i.item_id) for k, v in sorted(s.items())}


def stratified_sample(items: list[Item], n: int, seed: int = 42, min_per: int = 0) -> list[Item]:
    """Stratify by qtype; within each stratum (sorted by item_id) draw with one seeded RNG."""
    strata = _strata(items)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n, min_per)
    rng = random.Random(seed)
    out: list[Item] = []
    for k, members in strata.items():
        out += rng.sample(members, alloc[k])
    return sorted(out, key=lambda i: i.item_id)


def _vid_ranks(vid: list[Item], seed: int) -> dict[str, int]:
    """Round-robin across qtypes, each qtype in seeded order, so any long prefix is ~stratified."""
    rng = random.Random(seed)
    queues = []
    for _, members in _strata(vid).items():
        members = members[:]
        rng.shuffle(members)
        queues.append(members)
    order: list[str] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0).item_id)
    return {iid: r for r, iid in enumerate(order)}


def assign_nested(items: list[Item], sizes: tuple[int, int, int] = (2000, 300, 200), seed: int = 42,
                  msvd_min_per: int = 15) -> list[Item]:
    """Set in_main / in_annotated / in_vid_subset / vid_rank per benchmark. Unavailable items are never
    sampled. The MSVD per-type minimum is enforced at the main and annotated levels."""
    n_main, n_ann, n_vid = sizes
    flags: dict[str, dict] = {}
    by_bench: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_bench[it.benchmark].append(it)
    for bench, members in sorted(by_bench.items()):
        min_per = msvd_min_per if bench == "msvd" else 0
        pool = [i for i in members if i.available]
        main = stratified_sample(pool, n_main, seed, min_per)
        ann = stratified_sample(main, n_ann, seed, min_per)
        vid = stratified_sample(ann, n_vid, seed)
        ranks = _vid_ranks(vid, seed)
        for i in main:
            flags.setdefault(i.item_id, {})["in_main"] = True
        for i in ann:
            flags[i.item_id]["in_annotated"] = True
        for i in vid:
            flags[i.item_id].update(in_vid_subset=True, vid_rank=ranks[i.item_id])
    return [it.model_copy(update={"in_main": False, "in_annotated": False, "in_vid_subset": False,
                                  "vid_rank": None, **flags.get(it.item_id, {})}) for it in items]


def split_dev_test(items: list[Item], frac_dev: float = 0.4, seed: int = 42) -> list[Item]:
    """Assign human_split to annotated items by question, stratified by qtype within each benchmark."""
    dev_ids: set[str] = set()
    ann_by_bench: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.in_annotated:
            ann_by_bench[it.benchmark].append(it)
    for bench, ann in sorted(ann_by_bench.items()):
        dev_ids |= {i.item_id for i in stratified_sample(ann, round(frac_dev * len(ann)), seed)}
    return [it.model_copy(update={"human_split": ("dev" if it.item_id in dev_ids else "test")
                                  if it.in_annotated else None}) for it in items]
