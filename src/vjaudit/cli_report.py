"""`vjaudit.cli report ...`: the results summary (judges vs reference labels, visual gain)."""
from __future__ import annotations

from pathlib import Path

import typer

from .cli_common import PROFILE_HELP, open_profile, profile_doc
from .paths import DEFAULT_PATHS_FILE

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Reports.")


@app.command("summary")
def summary_cmd(out: Path = typer.Option(None, help="default: docs/results_<profile>.md"),
                paths: Path = typer.Option(DEFAULT_PATHS_FILE),
                profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Write the results tables for run 1 (annotated set) and run 2 (main set)."""
    from .report.summary import write_summary
    from .schemas import Item

    prof = open_profile(profile, paths)
    out = out or profile_doc(prof, "results")
    store = prof.store()
    write_summary(store, [Item(**r) for r in store.iter("item")], prof.name, out)
    typer.echo(f"wrote {out}")
