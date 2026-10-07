"""`vjaudit.cli annotate ...`: export label batches (for Claude or people), import labels, show progress."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from .cli_common import PROFILE_HELP, open_profile
from .paths import DEFAULT_PATHS_FILE

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Labelling: export batches, import labels.")
OUT_HELP = "base folder; files go to <out-dir>/<profile>/<claude|human>/"


def _audience(source: str) -> str:
    return "human" if source.startswith("human:") else "claude"


@app.command("export")
def export_cmd(set_: str = typer.Option("annotated", "--set", help="annotated | main"),
               for_: str = typer.Option("claude", "--for", help="claude (Markdown to paste) | human (CSV sheets)"),
               batch_size: int = typer.Option(100), out_dir: Path = typer.Option(Path("annotation"), help=OUT_HELP),
               paths: Path = typer.Option(DEFAULT_PATHS_FILE),
               profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Build answer units for the set, store them, and write the batches not exported yet."""
    from .annotate.export import export_claude, export_human
    from .annotate.units import build_units
    from .generate.coverage import select
    from .schemas import Item

    if set_ not in ("annotated", "main") or for_ not in ("claude", "human"):
        raise typer.BadParameter("--set is annotated|main and --for is claude|human")
    prof = open_profile(profile, paths)
    store = prof.store()
    items = select([Item(**r) for r in store.iter("item")], set_)
    units = build_units(items, list(store.iter("generation")))
    if not units:
        typer.echo("no answers to label yet: run generation first (make run1)")
        raise typer.Exit(1)
    for u in units:
        store.put("unit", u, u.unit_id)
    folder = out_dir / prof.name / for_
    files = (export_claude if for_ == "claude" else export_human)(units, items, folder, batch_size)
    typer.echo(f"{len(units)} units for {len(items)} questions; {len(files)} new batch file(s) in {folder}")
    for f in files:
        typer.echo(f"  {f}")
    if for_ == "claude" and files:
        typer.echo("Next: give each batch_NN.md to Claude, save its reply (e.g. batch_01_reply.txt), then run\n"
                   "  uv run python -m vjaudit.cli annotate import --source claude-manual --file <reply file>")


@app.command("import")
def import_cmd(source: str = typer.Option(..., help="claude-manual[-suffix] or human:<annotator-id>"),
               file: list[Path] = typer.Option(..., "--file", help="Claude reply or filled CSV; repeat for more"),
               out_dir: Path = typer.Option(Path("annotation"), help=OUT_HELP),
               paths: Path = typer.Option(DEFAULT_PATHS_FILE),
               profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Validate every row of every file; store the labels only if all rows are valid."""
    from .annotate.export import load_manifest
    from .annotate.importer import LabelImportError, import_labels, parse_label_csv, validate_source

    try:
        validate_source(source)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(2) from e
    prof = open_profile(profile, paths)
    manifest = load_manifest(out_dir / prof.name / _audience(source))
    if not manifest:
        typer.echo(f"no manifest in {out_dir / prof.name / _audience(source)}: run `annotate export` first")
        raise typer.Exit(1)
    rows, errors = [], []
    for f in file:
        try:
            parsed = parse_label_csv(f.read_text(encoding="utf-8"))
        except LabelImportError as e:
            errors += [f"{f.name}: {m}" for m in e.errors]
            continue
        rows += [{**r, "row": f"{f.name} row {r['row']}"} for r in parsed]
    store = prof.store()
    try:
        if errors:
            raise LabelImportError(errors)
        res = import_labels(store, rows, source, manifest)
    except LabelImportError as e:
        for m in e.errors:
            typer.echo(m)
        typer.echo(f"{len(e.errors)} problem(s); nothing was stored. Fix the file(s) and import again.")
        raise typer.Exit(1) from e
    typer.echo(f"stored {res.stored} label(s) from {source}; {res.skipped_existing} already present (skipped)")
    for b, k in sorted(res.missing_by_batch.items()):
        typer.echo(f"  {b}: {k} unit(s) still missing")


@app.command("status")
def status_cmd(out_dir: Path = typer.Option(Path("annotation"), help=OUT_HELP),
               paths: Path = typer.Option(DEFAULT_PATHS_FILE),
               profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Per exported batch: units, labels per source, and units with no label from any source."""
    from .annotate.export import load_manifest

    prof = open_profile(profile, paths)
    labels = list(prof.store().iter("label"))
    for audience in ("claude", "human"):
        manifest = load_manifest(out_dir / prof.name / audience)
        if not manifest:
            continue
        typer.echo(f"\n{audience} batches ({out_dir / prof.name / audience})")
        typer.echo("| batch | units | labelled | missing |\n|---|---|---|---|")
        for b, ids in manifest.items():
            idset = set(ids)
            per_source = Counter(r["source"] for r in labels if r["unit_id"] in idset)
            done = ", ".join(f"{s}: {k}" for s, k in sorted(per_source.items())) or "—"
            missing = len(idset - {r["unit_id"] for r in labels})
            typer.echo(f"| {b} | {len(ids)} | {done} | {missing} |")
