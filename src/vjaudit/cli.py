"""Command-line entry point: ``python -m vjaudit.cli <group> <command>``."""
from __future__ import annotations

import typer

from .cli_annotate import app as annotate_app
from .cli_data import app as data_app
from .cli_gen import app as gen_app
from .cli_judge import app as judge_app
from .cli_report import app as report_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(data_app, name="data")
app.add_typer(gen_app, name="gen")
app.add_typer(judge_app, name="judge")
app.add_typer(annotate_app, name="annotate")
app.add_typer(report_app, name="report")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("vjaudit 0.1.0")


if __name__ == "__main__":
    app()
