"""`vjaudit.cli gen ...` commands (generation)."""
from __future__ import annotations

from pathlib import Path

import typer

from .cli_common import PROFILE_HELP, open_profile
from .paths import DEFAULT_PATHS_FILE
from .profile import profile_root

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Generation: pinning, runs, coverage.")


@app.command("pin-revisions")
def pin_revisions_cmd(path: Path = typer.Option(None, help="default: <profile>/generators.yaml"),
                      profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Resolve each generator's current HF revision SHA and write it into generators.yaml."""
    from .generate.pin import pin_revisions

    root = profile_root(profile)
    typer.echo(f"profile: {root.name}")
    for gid, sha in pin_revisions(path or root / "generators.yaml").items():
        typer.echo(f"{gid}: {sha}")


@app.command("render-golden")
def render_golden_cmd(out_dir: Path = typer.Option(Path("tests/generate/golden"))) -> None:
    """Render the golden prompts from the pinned processors (inspect them by eye, then commit)."""
    from .generate.adapters import golden_render

    out_dir.mkdir(parents=True, exist_ok=True)
    for family in ("qwen-vl", "internvl", "llava-next-video"):
        for mode in ("video", "blind"):
            text = golden_render(family, mode)
            (out_dir / f"{family}_{mode}.txt").write_text(text, encoding="utf-8")
            typer.echo(f"--- {family} / {mode} ---\n{text}\n")


def _load(profile: str | None, paths: Path):
    from .schemas import Item

    prof = open_profile(profile, paths)
    store = prof.store()
    items = [Item(**r) for r in store.iter("item")]
    return prof, store, items, prof.generators


@app.command("run")
def run_cmd(subset: str = typer.Option(..., help="annotated | main"), mc: bool = typer.Option(False),
            shard: int = typer.Option(0), num_shards: int = typer.Option(1), batch_size: int = typer.Option(16),
            paths: Path = typer.Option(DEFAULT_PATHS_FILE),
            profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Generate every missing (item, system) for this shard's generators (vLLM, GPU)."""
    from .data.frames import FrameCache
    from .generate.adapters import HFChatRenderer
    from .generate.coverage import run_subset
    from .generate.engine import VllmEngine

    prof, store, items, gens = _load(profile, paths)
    n = run_subset(store, items, gens, subset, engine_factory=VllmEngine,
                   renderer_factory=lambda g: HFChatRenderer(g.hf_repo, g.revision, g.family),
                   frame_cache=FrameCache(prof.frame_cache), mc=mc, shard=shard, num_shards=num_shards,
                   batch_size=batch_size)
    typer.echo(f"shard {shard}/{num_shards}: {n} new generations")


@app.command("coverage")
def coverage_cmd(subset: str = typer.Option(...), mc: bool = typer.Option(False),
                 paths: Path = typer.Option(DEFAULT_PATHS_FILE),
                 profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Print missing (item, system) pairs per benchmark and system."""
    from .generate.coverage import coverage, select
    from .generate.systems import expand_mc_systems, expand_systems

    _, store, items, gens = _load(profile, paths)
    chosen = select(items, subset, mc)
    total = 0
    for b in sorted({i.benchmark for i in chosen}):
        systems = [s.system_id for s in (expand_mc_systems(gens) if mc else expand_systems(gens, b))]
        for sid, missing in coverage(store, [i for i in chosen if i.benchmark == b], systems).items():
            typer.echo(f"{b} {sid} missing {missing}")
            total += missing
    typer.echo(f"TOTAL missing {total}")


@app.command("rerun-check")
def rerun_check_cmd(system: str = typer.Option(...), n: int = typer.Option(50), benchmark: str = typer.Option("msvd"),
                    paths: Path = typer.Option(DEFAULT_PATHS_FILE),
                    profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Regenerate n items for one system into a scratch store and compare outputs (SP2 acceptance)."""
    import tempfile

    from .data.frames import FrameCache
    from .generate.adapters import HFChatRenderer, get_adapter
    from .generate.coverage import rerun_check
    from .generate.engine import VllmEngine
    from .generate.systems import expand_systems

    prof, store, items, gens = _load(profile, paths)
    spec = next(s for s in expand_systems(gens, benchmark) if s.system_id == system)
    gen = next(g for g in gens if g.id == spec.generator_id)
    adapter = get_adapter(gen.family, HFChatRenderer(gen.hf_repo, gen.revision, gen.family))
    same, total, mism = rerun_check(store, [i for i in items if i.benchmark == benchmark], spec, VllmEngine(gen),
                                    FrameCache(prof.frame_cache), gen, adapter, n, Path(tempfile.mkdtemp()))
    typer.echo(f"{same}/{total} identical")
    for iid, old, new in mism:
        typer.echo(f"MISMATCH {iid}: {old!r} != {new!r}")
    raise typer.Exit(0 if same == total else 1)


@app.command("mc-report")
def mc_report_cmd(out: Path = typer.Option(Path("docs/mc_anchor.md")),
                  paths: Path = typer.Option(DEFAULT_PATHS_FILE),
                  profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Write the judge-free NExT MC-anchor accuracy table."""
    from .generate.coverage import mc_accuracy
    from .generate.systems import expand_mc_systems

    _, store, items, gens = _load(profile, paths)
    acc = mc_accuracy(store, items, [s.system_id for s in expand_mc_systems(gens)])
    lines = ["# NExT-QA multiple-choice anchor (exact match, no judge)", "",
             "| system | n | accuracy | unparseable |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['n']} | {v['acc']:.3f} | {v['unparseable']} |" for s, v in sorted(acc.items())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}")
