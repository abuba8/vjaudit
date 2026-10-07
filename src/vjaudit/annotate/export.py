"""Write label batches: Markdown files to paste into Claude, or CSV sheets for people.

Exports never show system ids, model names or judge verdicts (R-ANN-02). Re-running an export only writes
units that are not in the folder's manifest yet, as new batches, so earlier batches and their labels stay valid.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from vjaudit.schemas import Item, Unit

from .units import plan_batches

MANIFEST = "manifest.json"

CLAUDE_HEADER = """# VideoQA answer labelling — batch {nn} of {total}

You are labelling answers to questions about short videos. You will NOT see the videos. For each
candidate answer, decide whether it is correct **given the question and the reference answer(s)**.

## Rules
1. Correct if it means the same as any reference answer, including synonyms and paraphrases.
2. More specific is fine ("puppy" for "dog"); more general is wrong ("animal" for "dog") unless the
   question asks for a category.
3. Partial answers to a multi-part reference are incorrect unless the missing part is not asked for.
4. Listing several alternatives ("a cat or a dog") is incorrect. Hedging around one answer: judge the answer.
5. Refusals, "cannot tell", and non-answers are incorrect.
6. Ignore extra details you cannot check, unless they contradict the reference.

## Error tags (for incorrect answers; optional for correct ones)
E1 synonym/paraphrase · E2 too general/too specific · E3 partial · E4 extra wrong detail ·
E5 shares words with the reference but means something else · E6 several alternatives ·
E7 refusal/non-answer · E8 reference looks wrong or ambiguous · E9 question ambiguous

## Output format
Reply with ONE csv code block and nothing else. Header exactly:
`unit_id,label,error_tags,note`
One row per unit below. label is `correct` or `incorrect`. error_tags separated by `;` (may be empty).
note is optional; quote it if it contains a comma.

## Items
"""


def load_manifest(out_dir: Path) -> dict[str, list[str]]:
    p = Path(out_dir) / MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _cell(text: str) -> str:
    """Safe inside a Markdown table cell: one line, escaped pipes, visible when empty."""
    s = _one_line(text).replace("|", "\\|")
    return s if s else "(empty answer)"


def _new_batches(units: list[Unit], out_dir: Path, batch_size: int) -> tuple[list[tuple[str, list[Unit]]], dict]:
    manifest = load_manifest(out_dir)
    done = {u for ids in manifest.values() for u in ids}
    pending = [u for u in units if u.unit_id not in done]
    start = len(manifest)
    named = [(f"batch_{start + k + 1:02d}", b) for k, b in enumerate(plan_batches(pending, batch_size))]
    return named, manifest


def _write_manifest(out_dir: Path, manifest: dict, named: list[tuple[str, list[Unit]]]) -> None:
    manifest = {**manifest, **{name: [u.unit_id for u in b] for name, b in named}}
    (Path(out_dir) / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def export_claude(units: list[Unit], items: list[Item], out_dir: Path, batch_size: int = 100) -> list[Path]:
    """Write batch_NN.md files (self-contained: task, rules, tags, output format, items) and the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {i.item_id: i for i in items}
    named, manifest = _new_batches(units, out_dir, batch_size)
    total = len(manifest) + len(named)
    paths = []
    for name, batch in named:
        lines = [CLAUDE_HEADER.format(nn=name.split("_")[1], total=f"{total:02d}")]
        q = 0
        for item_id in dict.fromkeys(u.item_id for u in batch):
            q += 1
            it = by_id[item_id]
            lines += [f"### Q{q}. {_one_line(it.question)}",
                      f"Reference answer(s): {' | '.join(_one_line(a) for a in it.answers_ref)}",
                      "| unit_id | candidate answer |", "|---|---|"]
            lines += [f"| {u.unit_id} | {_cell(u.display_answer)} |" for u in batch if u.item_id == item_id]
            lines.append("")
        p = out_dir / f"{name}.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        paths.append(p)
    if named:
        _write_manifest(out_dir, manifest, named)
    return paths


HUMAN_COLUMNS = ["unit_id", "benchmark", "question", "reference_answers", "candidate_answer", "label",
                 "error_tags", "note"]


def export_human(units: list[Unit], items: list[Item], out_dir: Path, batch_size: int = 100) -> list[Path]:
    """Write batch_NN.csv sheets with empty label columns, plus the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {i.item_id: i for i in items}
    named, manifest = _new_batches(units, out_dir, batch_size)
    paths = []
    for name, batch in named:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(HUMAN_COLUMNS)
        for u in batch:
            it = by_id[u.item_id]
            w.writerow([u.unit_id, u.benchmark, _one_line(it.question), " | ".join(it.answers_ref),
                        _one_line(u.display_answer), "", "", ""])
        p = out_dir / f"{name}.csv"
        p.write_text(buf.getvalue(), encoding="utf-8")
        paths.append(p)
    if named:
        _write_manifest(out_dir, manifest, named)
    return paths
