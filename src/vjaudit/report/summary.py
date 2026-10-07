"""Results summary: accuracy per system under each judge config and label source, agreement, visual gain."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from vjaudit.generate.coverage import select
from vjaudit.hashing import short_id
from vjaudit.schemas import Item
from vjaudit.store import ArtifactStore

from .metrics import cohen_kappa, error_rates, yes_rate

NOTE = "Label sources other than `human:*` are reference judges, not human ground truth."
DASH = "—"


def _config_key(call: dict) -> str:
    template = call["protocol_id"].split("@")[0]
    decoding = call["protocol_id"].split("/")[-1]
    return f"{call['judge_id']}/{template}/{decoding}"


def _primary(call: dict) -> bool:
    return call.get("sample_idx", 0) == 0 and call.get("input_mode", "text") == "text"


def judge_configs(judgecalls) -> list[str]:
    """Sorted '<judge_id>/<template>/<decoding_tag>' keys over first-sample, text-only calls."""
    return sorted({_config_key(c) for c in judgecalls if _primary(c)})


def build_tables(store: ArtifactStore, items: list[Item], subset: str) -> dict:
    """{benchmark: {system: {n, configs: {cfg: {acc, unparseable}}, sources: {src: {acc, n}},
    agreement: {cfg: {source, n, gap_pts, kappa, fpr, fnr}}}}} for the generations of ``subset``."""
    chosen = {i.item_id: i for i in select(items, subset)}
    calls = [c for c in store.iter("judgecall") if _primary(c)]
    verdict: dict[str, dict[str, bool | None]] = defaultdict(dict)
    for c in calls:
        verdict[c["gen_id"]][_config_key(c)] = c["parsed_verdict"]
    labels: dict[str, dict[str, bool]] = defaultdict(dict)
    for lab in store.iter("label"):
        if lab["task"] == "ref":
            labels[lab["unit_id"]][lab["source"]] = lab["label"] == "correct"
    configs = judge_configs(calls)
    sources = sorted({s for per_unit in labels.values() for s in per_unit})
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for g in store.iter("generation"):
        if g["item_id"] in chosen and g.get("style") != "mc":
            groups[chosen[g["item_id"]].benchmark][g["system_id"]].append(g)
    tables: dict = {}
    for bench in sorted(groups):
        tables[bench] = {}
        for system in sorted(groups[bench]):
            gens = groups[bench][system]
            unit = {g["gen_id"]: short_id(g["item_id"], g["answer_norm"]) for g in gens}
            row = {"n": len(gens), "configs": {}, "sources": {}, "agreement": {}}
            for cfg in configs:
                vs = [verdict[g["gen_id"]][cfg] for g in gens if cfg in verdict[g["gen_id"]]]
                if vs:
                    parsed = [v for v in vs if v is not None]
                    row["configs"][cfg] = {"acc": yes_rate(parsed), "unparseable": len(vs) - len(parsed)}
            for src in sources:
                ls = [labels[unit[g["gen_id"]]][src] for g in gens if src in labels.get(unit[g["gen_id"]], {})]
                if ls:
                    row["sources"][src] = {"acc": yes_rate(ls), "n": len(ls)}
            if sources:
                ref_src = sources[0]
                for cfg in configs:
                    pairs = [(verdict[g["gen_id"]][cfg], labels[unit[g["gen_id"]]][ref_src]) for g in gens
                             if verdict[g["gen_id"]].get(cfg) is not None
                             and ref_src in labels.get(unit[g["gen_id"]], {})]
                    if pairs:
                        j, r = [p[0] for p in pairs], [p[1] for p in pairs]
                        rates = error_rates(j, r)
                        row["agreement"][cfg] = {"source": ref_src, "n": len(pairs),
                                                 "gap_pts": 100 * (yes_rate(j) - yes_rate(r)),
                                                 "kappa": cohen_kappa(j, r), "fpr": rates["fpr"], "fnr": rates["fnr"]}
            tables[bench][system] = row
    return tables


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def visual_gain(tables: dict) -> dict:
    """Δ = acc(S) − acc(S-blind) per judge config and per label source, for every S with a blind partner."""
    out: dict = {}
    for bench, rows in tables.items():
        for system, row in rows.items():
            blind = rows.get(f"{system}-blind")
            if blind is None:
                continue
            out.setdefault(bench, {})[system] = {
                "configs": {c: _delta(v["acc"], blind["configs"][c]["acc"])
                            for c, v in row["configs"].items() if c in blind["configs"]},
                "sources": {s: _delta(v["acc"], blind["sources"][s]["acc"])
                            for s, v in row["sources"].items() if s in blind["sources"]}}
    return out


def _f(x: float | None, fmt: str = ".3f") -> str:
    return DASH if x is None else format(x, fmt)


def _all_keys(rows: dict, field: str) -> list[str]:
    return sorted({k for r in rows.values() for k in r[field]})


def _accuracy_table(rows: dict, with_sources: bool) -> list[str]:
    configs = _all_keys(rows, "configs")
    sources = _all_keys(rows, "sources") if with_sources else []
    head = ["system", "n"] + configs + sources
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for system, r in rows.items():
        cells = [system, str(r["n"])]
        cells += [f"{_f(r['configs'][c]['acc'])} ({r['configs'][c]['unparseable']})" if c in r["configs"] else DASH
                  for c in configs]
        cells += [_f(r["sources"][s]["acc"]) if s in r["sources"] else DASH for s in sources]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_summary(store: ArtifactStore, items: list[Item], profile_name: str, out: Path) -> None:
    """Write docs/results_<profile>.md: run-1 tables (annotated), visual gain, run-2 table (main)."""
    run1 = build_tables(store, items, "annotated")
    lines = [f"# Results — profile `{profile_name}`", "", f"> {NOTE}",
             "> acc = share of answers called correct. (k) after a judge accuracy = verdicts that could not be "
             "parsed; they are counted, never guessed.", "", "## Run 1 — annotated set", ""]
    if not run1:
        lines += ["No generations for the annotated set yet.", ""]
    for bench, rows in run1.items():
        lines += [f"### {bench}", ""] + _accuracy_table(rows, with_sources=True) + [""]
        agree = [(s, c, a) for s, r in rows.items() for c, a in sorted(r["agreement"].items())]
        if not agree:
            lines += ["Agreement: no labels imported yet (see docs/EXPLAINED.md, 'Labelling with Claude').", ""]
            continue
        lines += [f"Agreement with `{agree[0][2]['source']}` (gap = judge acc − reference acc, in points):", "",
                  "| system | judge config | n | gap (pts) | κ | FPR | FNR |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {s} | {c} | {a['n']} | {_f(a['gap_pts'], '+.1f')} | {_f(a['kappa'])} | {_f(a['fpr'])} | "
                  f"{_f(a['fnr'])} |" for s, c, a in agree]
        lines.append("")
    gain = visual_gain(run1)
    lines += ["## Visual gain (annotated set)", "",
              "Δ = accuracy with video − accuracy of the same model answering blind (no video).", ""]
    if not gain:
        lines += ["No system has a blind partner with results yet.", ""]
    for bench, per_system in gain.items():
        for system, d in per_system.items():
            parts = [f"{c}: {_f(v, '+.3f')}" for c, v in sorted(d["configs"].items())]
            parts += [f"{s}: {_f(v, '+.3f')}" for s, v in sorted(d["sources"].items())]
            lines.append(f"- {bench} / {system}: " + ("; ".join(parts) or DASH))
    lines += ["", "## Run 2 — main set", ""]
    run2 = build_tables(store, items, "main")
    if not run2:
        lines += ["No generations for the main set yet.", ""]
    for bench, rows in run2.items():
        lines += [f"### {bench}", ""] + _accuracy_table(rows, with_sources=False) + [""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
