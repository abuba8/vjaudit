"""Helpers to test notebooks/pipeline.ipynb: exec its definition cells, or run it end to end."""
from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "pipeline.ipynb"


def read_notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK, as_version=4)


def tags(cell) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []))


def load_lib(params: dict | None = None) -> dict:
    """Exec the `parameters` cell (with overrides) and every `lib` cell into one namespace."""
    ns: dict = {"__name__": "pipeline_notebook"}
    for cell in read_notebook().cells:
        if cell.cell_type != "code":
            continue
        if "parameters" in tags(cell):
            exec(cell.source, ns)
            ns.update(params or {})
        elif "lib" in tags(cell):
            exec(cell.source, ns)
    return ns


@pytest.fixture(scope="module")
def nb_lib() -> dict:
    return load_lib()
