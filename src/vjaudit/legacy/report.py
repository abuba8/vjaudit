"""Write docs/sp0_report.md: identity tables (R-SP0-01), Tables 2–3 vs expected (R-SP0-04), S1 label (R-SP0-05)."""
from __future__ import annotations

import json
from pathlib import Path

from .diff import pairwise_identity
from .load import load_legacy_cells, load_legacy_config, s1_one_shot_labels
from .tables import table2, table3


def _flat3(t: dict) -> dict[tuple, float]:
    return {(ds, c, fr, s): v for ds, cs in t.items() for c, frs in cs.items()
            for fr, ss in frs.items() for s, v in ss.items()}


def write_sp0_report(cfg_path: Path, expected_path: Path, out: Path) -> None:
    cfg = load_legacy_config(cfg_path)
    cells = load_legacy_cells(cfg_path)
    expected = json.loads(Path(expected_path).read_text())
    lines = ["# SP0 report — prior pipeline integrity", "",
             f"Legacy quantization: {cfg.get('legacy_quantization', 'unknown')}", ""]
    for split in ("test", "val"):
        df = pairwise_identity(cells, split)
        lines += [f"## Generation identity across conditions ({split})", "",
                  df.to_markdown(index=False) if len(df) else "_no cells for this split_", ""]
    got3 = _flat3(table3(cells))
    lines += ["## Table 3 reproduction", "", "| dataset | cond | frames | set | expected | got | Δ |", "|---|---|---|---|---|---|---|"]
    for key, exp in sorted(_flat3(expected["table3"]).items()):
        got = got3.get(key)
        delta = "missing" if got is None else f"{got - exp:+.2f}"
        lines.append(f"| {' | '.join(map(str, key))} | {exp:.2f} | {'' if got is None else f'{got:.2f}'} | {delta} |")
    lines += ["", "## Table 2 reproduction", ""]
    if cfg.get("consensus"):
        got2 = table2(cells, cfg["consensus"])
        lines += ["| dataset | set | expected J/FP/FN | got J/FP/FN |", "|---|---|---|---|"]
        for ds, rows in expected["table2"].items():
            for s in ("S2", "S3", "S4", "S5"):
                e = rows[s]
                g = got2.get(ds, {}).get(s)
                gtxt = "missing" if g is None else f"{g['J']:.1f} / {g['FP']} / {g['FN']}"
                lines.append(f"| {ds} | {s} | {e['J']:.1f} / {e['FP']} / {e['FN']} | {gtxt} |")
    else:
        lines.append("_no consensus annotations configured_")
    labels = s1_one_shot_labels(Path(cfg["prompts_dir"]))
    lines += ["", "## R-SP0-05", "", f"S1 one-shot demonstration verdict label(s): {', '.join(labels) or 'not found'}", "",
              "## R-SP0-02 (manual)", "", "See docs/decisions.md for the investigation outcome (a / b / c).", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
