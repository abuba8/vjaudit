"""R-SP0-04: reproduce BlackboxNLP Tables 2–3 from cached judge outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .load import LegacyCell


def _is_yes(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == "yes"


def acceptance(judge_json: dict) -> float:
    """100 × Yes / total. Anything other than Yes (incl. unparseable) counts as No, as in the prior paper."""
    vals = [row.get("Similarity") for row in judge_json.values()]
    return round(100.0 * sum(_is_yes(v) for v in vals) / len(vals), 2) if vals else float("nan")


def table3(cells: list[LegacyCell]) -> dict:
    """{dataset: {cond: {frames: {"zero"|set: acceptance}}}} on the test split."""
    out: dict = {}
    zero = {(c.dataset, c.frames): acceptance(c.judge()) for c in cells
            if c.split == "test" and c.condition == "zero"}
    for c in cells:
        if c.split != "test" or c.condition == "zero":
            continue
        slot = out.setdefault(c.dataset, {}).setdefault(c.condition, {}).setdefault(str(c.frames), {})
        slot[c.set] = acceptance(c.judge())
        if (c.dataset, c.frames) in zero:
            slot["zero"] = zero[(c.dataset, c.frames)]
    return out


def load_consensus(spec: dict) -> dict[str, bool]:
    """Human-consensus labels keyed by legacy index; ``spec`` = {path, key_field, label_field, yes_values}."""
    path = Path(spec["path"])
    yes = {str(v).strip().lower() for v in spec["yes_values"]}
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
        rows = df.to_dict("records")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [{spec["key_field"]: k, **v} for k, v in data.items()] if isinstance(data, dict) else data
    return {str(r[spec["key_field"]]): str(r[spec["label_field"]]).strip().lower() in yes for r in rows}


def table2(cells: list[LegacyCell], consensus_specs: dict) -> dict:
    """Few-shot judge vs human consensus on the annotated validation sample: J%, FP, FN per set."""
    out: dict = {}
    for ds, spec in consensus_specs.items():
        cons = load_consensus(spec)
        out[ds] = {"consensus": round(100.0 * sum(cons.values()) / len(cons), 2)}
        for c in cells:
            if c.dataset != ds or c.split != "val" or c.condition != "few":
                continue
            judge = c.judge()
            keys = [k for k in cons if k in judge]
            verdict = {k: _is_yes(judge[k].get("Similarity")) for k in keys}
            out[ds][c.set] = {
                "J": round(100.0 * sum(verdict.values()) / len(keys), 2),
                "FP": sum(verdict[k] and not cons[k] for k in keys),
                "FN": sum((not verdict[k]) and cons[k] for k in keys),
            }
    return out
