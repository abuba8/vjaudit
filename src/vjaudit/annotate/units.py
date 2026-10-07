"""Units: one per (question, normalized answer), so identical answers from several systems are labelled once."""
from __future__ import annotations

from collections import defaultdict

from vjaudit.hashing import short_id
from vjaudit.schemas import Item, Unit
from vjaudit.store import stamp

__all__ = ["Unit", "build_units", "plan_batches"]


def build_units(items: list[Item], generations: list[dict]) -> list[Unit]:
    """Group the open-ended generations of ``items`` by (item_id, answer_norm); multiple-choice answers are skipped."""
    by_item = {i.item_id: i for i in items}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in generations:
        if g["item_id"] in by_item and g.get("style") != "mc":
            groups[(g["item_id"], g["answer_norm"])].append(g)
    units = []
    for (item_id, norm), gens in groups.items():
        gens = sorted(gens, key=lambda g: g["gen_id"])
        rec = dict(unit_id=short_id(item_id, norm), item_id=item_id, benchmark=by_item[item_id].benchmark,
                   answer_norm=norm, display_answer=gens[0]["output_text"], gen_ids=[g["gen_id"] for g in gens],
                   system_ids=sorted({g["system_id"] for g in gens}))
        units.append(Unit(**stamp(rec, {"unit_rule": "item_id|answer_norm"})))
    return sorted(units, key=lambda u: (u.item_id, u.unit_id))


def plan_batches(units: list[Unit], batch_size: int = 100) -> list[list[Unit]]:
    """Consecutive batches of about ``batch_size`` units; all units of one question stay in one batch (R-ANN-08)."""
    questions: dict[str, list[Unit]] = {}
    for u in units:
        questions.setdefault(u.item_id, []).append(u)
    batches: list[list[Unit]] = []
    current: list[Unit] = []
    for q_units in questions.values():
        if current and len(current) + len(q_units) > batch_size:
            batches.append(current)
            current = []
        current = current + q_units
    if current:
        batches.append(current)
    return batches
