"""`vjaudit.cli data ...` commands."""
from __future__ import annotations

from pathlib import Path

import typer

from .config import BenchmarksFile, GridCfg, load_yaml
from .data.availability import check_availability
from .data.loaders import load_benchmark
from .paths import DEFAULT_PATHS_FILE, load_paths
from .store import ArtifactStore

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Data loading, availability and sampling.")


@app.command("check-availability")
def check_availability_cmd(
    paths: Path = typer.Option(DEFAULT_PATHS_FILE, help="paths.yaml"),
    benchmarks: Path = typer.Option(Path("configs/benchmarks.yaml")),
    report: Path = typer.Option(Path("docs/data_availability.md")),
    grid: Path = typer.Option(Path("configs/grids/stage1.yaml")),
) -> None:
    """Load every benchmark, resolve + probe its videos, store the pool as kind 'item_pool'."""
    p = load_paths(paths)
    cfg = load_yaml(benchmarks, BenchmarksFile)
    store = ArtifactStore(p["artifacts"])
    items = []
    for b in cfg.benchmarks:
        loaded = load_benchmark(b.id, p[b.raw_dir_key], b.split)
        items += check_availability(loaded, p[b.video_dir_key], report_path=report.with_suffix(f".{b.id}.md"),
                                    exts=b.video_exts)
    for it in items:
        store.put("item_pool", it, it.item_id)
    _merge_reports(report, [report.with_suffix(f".{b.id}.md") for b in cfg.benchmarks])
    n_ok = sum(i.available for i in items)
    main_n = load_yaml(grid, GridCfg).sample_sizes["main"]
    for b in cfg.benchmarks:
        n_b = sum(i.available for i in items if i.benchmark == b.id)
        if n_b < main_n:
            typer.echo(f"WARNING: {b.id} has {n_b} available items < main sample {main_n}")
    typer.echo(f"{len(items)} items, {n_ok} available; report: {report}")


def _merge_reports(out: Path, parts: list[Path]) -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in parts if p.exists())
    out.write_text(text, encoding="utf-8")
    for p in parts:
        p.unlink(missing_ok=True)


@app.command("sample")
def sample_cmd(
    paths: Path = typer.Option(DEFAULT_PATHS_FILE),
    grid: Path = typer.Option(Path("configs/grids/stage1.yaml")),
    report: Path = typer.Option(Path("docs/samples.md")),
) -> None:
    """Draw main ⊃ annotated ⊃ vid samples and the dev/test split; store kind 'item'."""
    from collections import Counter

    from .data.sampling import assign_nested, split_dev_test
    from .schemas import Item

    g = load_yaml(grid, GridCfg)
    store = ArtifactStore(load_paths(paths)["artifacts"])
    pool = [Item(**r) for r in store.iter("item_pool")]
    s = g.sample_sizes
    items = split_dev_test(assign_nested(pool, (s["main"], s["annotated"], s["vid"]), g.seed, g.msvd_min_per),
                           g.frac_dev, g.seed)
    for it in items:
        store.put("item", it, it.item_id)
    lines = ["# Samples", "", f"seed {g.seed}; sizes {s}", "",
             "| benchmark | pool | main | annotated | vid | dev | test |", "|---|---|---|---|---|---|---|"]
    for b in sorted({i.benchmark for i in items}):
        bi = [i for i in items if i.benchmark == b]
        lines.append(f"| {b} | {len(bi)} | {sum(i.in_main for i in bi)} | {sum(i.in_annotated for i in bi)} | "
                     f"{sum(i.in_vid_subset for i in bi)} | {sum(i.human_split == 'dev' for i in bi)} | "
                     f"{sum(i.human_split == 'test' for i in bi)} |")
    lines += ["", "## Annotated questions per type", "", "| benchmark | qtype | annotated |", "|---|---|---|"]
    for (b, q), k in sorted(Counter((i.benchmark, i.qtype) for i in items if i.in_annotated).items()):
        lines.append(f"| {b} | {q} | {k} |")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"wrote {len(items)} items; report: {report}")
