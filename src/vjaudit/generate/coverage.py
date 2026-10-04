"""Run subsets of systems, check coverage and reproducibility, score the MC anchor."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from vjaudit.config import GeneratorCfg
from vjaudit.schemas import Item
from vjaudit.store import ArtifactStore

from .adapters import Adapter, get_adapter
from .engine import Engine
from .runner import run_generation
from .systems import SystemSpec, expand_mc_systems, expand_systems, parse_mc_letter


def _pairs(store: ArtifactStore) -> set[tuple[str, str]]:
    return {(r["item_id"], r["system_id"]) for r in store.iter("generation")}


def coverage(store: ArtifactStore, items: list[Item], systems: list[str]) -> dict[str, int]:
    """Missing (item, system) pairs per system, over available items."""
    have = _pairs(store)
    avail = [i for i in items if i.available]
    return {s: sum((i.item_id, s) not in have for i in avail) for s in systems}


def select(items: list[Item], subset: str, mc: bool = False) -> list[Item]:
    flag = {"annotated": "in_annotated", "main": "in_main"}[subset]
    out = [i for i in items if getattr(i, flag) and i.available]
    if mc:
        out = [i for i in out if i.benchmark == "nextoe" and i.mc_options]
    return out


def run_subset(store: ArtifactStore, items: list[Item], gens: list[GeneratorCfg], subset: str,
               engine_factory: Callable[[GeneratorCfg], Engine], renderer_factory: Callable[[GeneratorCfg], object],
               frame_cache, mc: bool = False, shard: int = 0, num_shards: int = 1, batch_size: int = 16) -> int:
    """Generate every missing (item, system) for this shard's generators; load each engine at most once."""
    chosen = select(items, subset, mc)
    benches = sorted({i.benchmark for i in chosen})
    have = _pairs(store)
    new = 0
    for k, g in enumerate(gens):
        if k % num_shards != shard:
            continue
        plan = []
        for b in benches:
            specs = expand_mc_systems([g]) if mc else expand_systems([g], b)
            b_items = [i for i in chosen if i.benchmark == b]
            plan += [(s, b_items) for s in specs if any((i.item_id, s.system_id) not in have for i in b_items)]
        if not plan:
            continue
        engine = engine_factory(g)
        adapter: Adapter = get_adapter(g.family, renderer_factory(g))
        for spec, b_items in plan:
            new += run_generation(store, b_items, spec, engine, frame_cache, g, adapter, batch_size)
    return new


def mc_accuracy(store: ArtifactStore, items: list[Item], systems: list[str]) -> dict[str, dict]:
    """Exact-match accuracy of the MC anchor; unparseable answers count as wrong and are reported."""
    gold = {i.item_id: i.mc_answer_idx for i in items if i.mc_answer_idx is not None}
    out = {s: {"n": 0, "correct": 0, "unparseable": 0} for s in systems}
    for r in store.iter("generation"):
        s = r["system_id"]
        if s in out and r["item_id"] in gold:
            pred = parse_mc_letter(r["output_text"])
            out[s]["n"] += 1
            out[s]["unparseable"] += pred is None
            out[s]["correct"] += pred == gold[r["item_id"]]
    for v in out.values():
        v["acc"] = v["correct"] / v["n"] if v["n"] else float("nan")
    return out


def rerun_check(store: ArtifactStore, items: list[Item], spec: SystemSpec, engine: Engine, frame_cache,
                gen: GeneratorCfg, adapter: Adapter, n: int, scratch: Path) -> tuple[int, int, list[tuple]]:
    """Regenerate ``n`` already-generated items into a scratch store; return (same, total, mismatches)."""
    old = {r["item_id"]: r for r in store.iter("generation") if r["system_id"] == spec.system_id}
    sample = sorted((i for i in items if i.item_id in old and i.available), key=lambda i: i.item_id)[:n]
    fresh = ArtifactStore(scratch)
    run_generation(fresh, sample, spec, engine, frame_cache, gen, adapter)
    new = {r["item_id"]: r for r in fresh.iter("generation")}
    mism = [(iid, old[iid]["output_text"], new[iid]["output_text"]) for iid in new
            if new[iid]["output_text"] != old[iid]["output_text"]]
    return len(new) - len(mism), len(new), mism
