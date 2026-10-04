"""`vjaudit.cli gen ...` commands (generation)."""
from __future__ import annotations

from pathlib import Path

import typer

from .paths import CONFIGS

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Generation: pinning, runs, coverage.")


@app.command("pin-revisions")
def pin_revisions_cmd(path: Path = typer.Option(CONFIGS / "generators.yaml")) -> None:
    """Resolve each generator's current HF revision SHA and write it into generators.yaml."""
    from .generate.pin import pin_revisions

    for gid, sha in pin_revisions(path).items():
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
