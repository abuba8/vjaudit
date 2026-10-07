"""Import labels from Claude's reply or a filled-in sheet: validate every row first, store only if all are valid.

Claude labels are a reference *judge* (source ``claude-manual[-suffix]``), never human labels (``human:<id>``).
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

from vjaudit.hashing import short_id
from vjaudit.schemas import Label
from vjaudit.store import ArtifactStore, stamp

SOURCE_RE = re.compile(r"^(claude-manual(-[A-Za-z0-9_.]+)?|human:[A-Za-z0-9_.-]+)$")
TAGS = {f"E{k}" for k in range(1, 10)}
LABELS = {"correct", "incorrect"}
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


class LabelImportError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class ImportResult:
    stored: int
    skipped_existing: int
    missing_by_batch: dict[str, int]


def validate_source(source: str) -> None:
    if not SOURCE_RE.match(source):
        raise ValueError(f"source {source!r} must be 'claude-manual', 'claude-manual-<suffix>' or "
                         "'human:<annotator-id>'")


def parse_label_csv(text: str) -> list[dict]:
    """Rows from a CSV, or from the first fenced code block that holds one (Claude's reply may add prose).

    Without a code fence, the CSV ends at the first non-empty line that has no comma (closing prose).
    Spaces after commas are ignored. A row with more fields than the header is flagged, not guessed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = [b for b in _FENCE.findall(text) if "unit_id" in b]
    body = blocks[0] if blocks else text
    lines = body.split("\n")
    start = next((k for k, ln in enumerate(lines) if ln.strip().lstrip("\ufeff").startswith("unit_id")), None)
    if start is None:
        raise LabelImportError(["no header line starting with unit_id (expected: unit_id,label,error_tags,note)"])
    kept = [lines[start].strip()]
    for ln in lines[start + 1:]:
        if not blocks and ln.strip() and "," not in ln:
            break  # prose after an unfenced CSV
        kept.append(ln.strip())
    reader = csv.DictReader(io.StringIO("\n".join(kept)), skipinitialspace=True, restkey="_extra")
    reader.fieldnames = [f.strip().lower() for f in reader.fieldnames or []]
    rows = []
    for n, r in enumerate(reader, start=1):
        if not any((v or "").strip() for k, v in r.items() if k != "_extra" and isinstance(v, str)):
            continue  # blank line
        tags = [t.strip().upper() for t in (r.get("error_tags") or "").split(";") if t.strip()]
        row = {"row": n, "unit_id": (r.get("unit_id") or "").strip(), "label": (r.get("label") or "").strip().lower(),
               "error_tags": tags, "note": (r.get("note") or "").strip()}
        if r.get("_extra"):
            row["too_many_fields"] = True
        rows.append(row)
    return rows


def _validate(rows: list[dict], batch_of: dict[str, str]) -> list[str]:
    errors, seen = [], {}
    for r in rows:
        at = r["row"] if isinstance(r["row"], str) else f"row {r['row']}"  # the CLI passes "<file> row <n>"
        where = f"{at} ({r['unit_id'] or 'no unit_id'})"
        if r.get("too_many_fields"):
            errors.append(f"{where}: more than 4 fields (quote a note that contains commas; separate tags with ;)")
        if r["unit_id"] not in batch_of:
            errors.append(f"{where}: unit_id {r['unit_id']!r} was not exported")
        if r["label"] not in LABELS:
            errors.append(f"{where}: label {r['label']!r} must be correct or incorrect")
        bad = [t for t in r["error_tags"] if t not in TAGS]
        if bad:
            errors.append(f"{where}: error tag(s) {', '.join(bad)} not in E1–E9")
        prev = seen.get(r["unit_id"])
        if prev is not None and prev != r["label"]:
            errors.append(f"{where}: unit {r['unit_id']} appears twice with different labels")
        seen.setdefault(r["unit_id"], r["label"])
    return errors


def import_labels(store: ArtifactStore, rows: list[dict], source: str, manifest: dict[str, list[str]],
                  task: str = "ref") -> ImportResult:
    """Validate all rows, then store Label records; existing (unit, source) labels are skipped, not replaced."""
    validate_source(source)
    batch_of = {u: b for b, ids in manifest.items() for u in ids}
    errors = _validate(rows, batch_of)
    if errors:
        raise LabelImportError(errors)
    stored = skipped = 0
    for r in rows:
        label_id = short_id(r["unit_id"], source)
        rec = Label(**stamp(dict(label_id=label_id, unit_id=r["unit_id"], source=source, task=task, label=r["label"],
                                 error_tags=r["error_tags"], note=r["note"], batch=batch_of[r["unit_id"]]),
                            {"source": source, "task": task}))
        if store.put("label", rec, label_id):
            stored += 1
        else:
            skipped += 1
    labelled = {r["unit_id"] for r in store.iter("label") if r["source"] == source}
    missing = {b: sum(u not in labelled for u in ids) for b, ids in manifest.items()}
    return ImportResult(stored, skipped, {b: k for b, k in missing.items() if k})
