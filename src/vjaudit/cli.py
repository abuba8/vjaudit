"""Command-line entry point: ``python -m vjaudit.cli <group> <command>``."""
from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("vjaudit 0.1.0")


if __name__ == "__main__":
    app()
