"""`vjaudit.cli judge ...` commands (templates provenance; GPT-3.5 capture is added in Task 13)."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml

from .judge.templates import JUDGE_DIR, vendor_rub, verify_vcg_source

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Judging: template provenance and capture.")
SOURCE = JUDGE_DIR / "source.yaml"


def _update_source(path: Path, template_id: str, **fields) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data[template_id].update(fields)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@app.command("verify-vcg")
def verify_vcg_cmd(upstream: list[Path] = typer.Option(..., help="upstream Video-ChatGPT eval script(s); first = ActivityNet"),
                   commit: str = typer.Option(..., help="upstream commit SHA the files came from"),
                   source: Path = typer.Option(SOURCE)) -> None:
    """Check t_vcg_*.txt against the upstream script(s); mark verified when the first file matches."""
    diffs = verify_vcg_source(upstream)
    for d in diffs:
        typer.echo(d)
    main_ok = not any(d.startswith(Path(upstream[0]).name + ":") for d in diffs)
    if main_ok:
        _update_source(source, "t-vcg", upstream_commit=commit, verified=True)
        typer.echo(f"t-vcg verified against {upstream[0]} @ {commit}")
    raise typer.Exit(0 if main_ok else 1)


@app.command("vendor-rub")
def vendor_rub_cmd(prior: Path = typer.Option(..., help="prior repo src/Prompts/LLM_as_judge_base.j2"),
                   var_map: str = typer.Option(..., help="prior Jinja var → slot, e.g. q=question,gt=answer,gen=pred"),
                   commit: str = typer.Option(...), out: Path = typer.Option(JUDGE_DIR / "t_rub.txt"),
                   source: Path = typer.Option(SOURCE)) -> None:
    """Replace t_rub.txt with the prior repo's template (zero-shot form) and mark it verified."""
    mapping = dict(kv.split("=", 1) for kv in var_map.split(","))
    if set(mapping.values()) != {"question", "answer", "pred"}:
        raise typer.BadParameter("var-map must map onto exactly question, answer, pred")
    text = vendor_rub(prior, mapping, out)
    _update_source(source, "t-rub", upstream_commit=commit, verified=True)
    typer.echo(text)
