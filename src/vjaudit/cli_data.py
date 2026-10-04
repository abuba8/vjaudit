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
