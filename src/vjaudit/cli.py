"""Command-line entry point: ``python -m vjaudit.cli <group> <command>``."""
from __future__ import annotations

import typer

from .cli_data import app as data_app
from .cli_gen import app as gen_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(data_app, name="data")
app.add_typer(gen_app, name="gen")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("vjaudit 0.1.0")


if __name__ == "__main__":
    app()
