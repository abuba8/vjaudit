"""Run the whole notebook in dry-run mode on synthetic data, for both profiles."""
import json
import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tests.notebook.conftest import NOTEBOOK, tags
from tests.test_e2e import _video
from vjaudit.data.loaders import load_benchmark

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "tests" / "fixtures" / "raw"
DIRS = {"msvd": "msvd", "anetqa": "anetqa", "nextoe": "nextqa"}


def make_world(tmp_path: Path) -> tuple[Path, Path]:
    """A fake repo (prompts only) and a data folder with the fixture questions and a small video for each."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "configs" / "prompts", repo / "configs" / "prompts")
    data = tmp_path / "data"
    for bench, folder in DIRS.items():
        shutil.copytree(RAW / folder, data / folder / "annotations")
        for it in load_benchmark(bench, data / folder / "annotations"):
            name = it.video_id if bench != "anetqa" else it.video_id
            path = data / folder / "videos" / f"{name}.mp4"
            if not path.exists():
                _video(path, n=20)
    (repo / "notebooks").mkdir()
    shutil.copy(NOTEBOOK, repo / "notebooks" / "pipeline.ipynb")   # where it lives in a real checkout
    return repo, data


def section_of_each_cell(nb) -> list[int]:
    out, cur = [], -1
    for c in nb.cells:
        m = re.search(r'<a id="s(\d+)"></a>', c.source) if c.cell_type == "markdown" else None
        cur = int(m.group(1)) if m else cur
        out.append(cur)
    return out


def run(repo: Path, params: dict, monkeypatch, sections: set | None = None) -> nbformat.NotebookNode:
    """Execute the notebook in a fresh kernel; with `sections`, only the cells of those sections."""
    monkeypatch.setenv("NOTEBOOK_REPO", str(repo))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in nb.cells if "parameters" in tags(c))
    for k, v in params.items():
        cell.source, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", cell.source, flags=re.M)
        assert n == 1, k
    if sections is not None:
        nb.cells = [c for c, s in zip(nb.cells, section_of_each_cell(nb)) if s in sections]
    NotebookClient(nb, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(repo / "notebooks")}}).execute()
    return nb


def errors(nb) -> list:
    return [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]



def test_mac_dry_run_end_to_end_then_import_labels(tmp_path, monkeypatch):
    repo, data = make_world(tmp_path)
    params = dict(PROFILE="mac", DRY_RUN=True, DATA_ROOT=str(data), RUN="annotated")
    nb = run(repo, params, monkeypatch)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
    runs = repo / "notebooks" / "runs" / "mac-dryrun"
    gens, calls = lines(runs / "generations.jsonl"), lines(runs / "judgecalls.jsonl")
    assert len(gens) == 9 * 5                       # 3 annotated questions × 3 benchmarks × 5 systems
    assert len(calls) == 4 * len(gens)              # stages M1–M4
    batch = runs / "annotation" / "claude" / "batch_01.md"
    assert batch.exists() and "unit_id,label,error_tags,note" in batch.read_text()
    assert "no labels imported yet" in (runs / "results.md").read_text()

    units = lines(runs / "units.jsonl")
    reply = "Here you go:\n```csv\nunit_id,label,error_tags,note\n" + "".join(
        f"{u['unit_id']},{'correct' if k % 2 == 0 else 'incorrect'},,\n" for k, u in enumerate(units)) + "```\n"
    (runs / "annotation" / "claude" / "replies" / "batch_01.txt").write_text(reply)
    run(repo, params, monkeypatch)                  # everything else is skipped; the import and report run
    assert len(lines(runs / "generations.jsonl")) == len(gens)
    assert len(lines(runs / "labels.jsonl")) == len(units)
    text = (runs / "results.md").read_text()
    assert "Agreement with `claude-manual`" in text and "## Visual gain" in text


def test_server_dry_run_end_to_end(tmp_path, monkeypatch):
    repo, data = make_world(tmp_path)
    nb = run(repo, dict(PROFILE="server", DRY_RUN=True, DATA_ROOT=str(data), RUN="annotated"), monkeypatch)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
    runs = repo / "notebooks" / "runs" / "server-dryrun"
    gens = lines(runs / "generations.jsonl")
    assert len(gens) == 9 * 11                      # 11 server systems
    stages = {c["stage"] for c in lines(runs / "judgecalls.jsonl")}
    assert stages == {"S1", "S2", "S5", "S6a", "S6b", "S7"}


@pytest.fixture(scope="module")
def finished_dry_run(tmp_path_factory):
    repo, data = make_world(tmp_path_factory.mktemp("restart"))
    params = dict(PROFILE="mac", DRY_RUN=True, DATA_ROOT=str(data), RUN="annotated")
    with pytest.MonkeyPatch.context() as mp:
        assert errors(run(repo, params, mp)) == []
    return repo, params


@pytest.mark.parametrize("section", range(4, 12))
def test_after_a_restart_sections_2_and_3_then_any_section(finished_dry_run, section, monkeypatch):
    repo, params = finished_dry_run
    nb = run(repo, params, monkeypatch, sections={2, 3, section})
    assert [f"{e['ename']}: {e['evalue']}" for e in errors(nb)] == []
