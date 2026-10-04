import yaml
from typer.testing import CliRunner

from tests.helpers.factories import make_generation, make_item
from tests.judge.test_capture import GROUPS, STAGES, _clients
from vjaudit.cli import app
from vjaudit.judge.capture import build_jobs, run_capture, write_capture_report
from vjaudit.store import ArtifactStore


def _paths(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    for i in range(2):
        it = make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True)
        store.put("item", it, it.item_id)
        g = make_generation(i, gen_id=f"g{i}", item_id=it.item_id, system_id="qwen3vl8b")
        store.put("generation", g, g.gen_id)
    pf = tmp_path / "paths.yaml"
    pf.write_text(yaml.safe_dump({"artifacts": str(tmp_path / "a"), "api_cache": str(tmp_path / "cache")}))
    return pf


def test_capture_refuses_unverified_templates(tmp_path):
    res = CliRunner().invoke(app, ["judge", "capture", "--paths", str(_paths(tmp_path)), "--stages", "S1"])
    assert res.exit_code == 2 and "t-vcg is not verified" in res.output


def test_capture_dry_run_prints_estimate_and_makes_no_calls(tmp_path):
    res = CliRunner().invoke(app, ["judge", "capture", "--paths", str(_paths(tmp_path)), "--stages", "S1,S2",
                                   "--allow-unverified"])
    assert res.exit_code == 0, res.output
    assert "S1: 2 jobs" in res.output and "S2: 2 jobs" in res.output and "dry run" in res.output
    assert not (tmp_path / "cache").exists()


async def test_capture_report_counts(tmp_path):
    from tests.judge.test_capture import _store
    store, items = _store(tmp_path)
    clients, _, _ = _clients(tmp_path)
    await run_capture(store, clients, build_jobs(store, STAGES[:1], items, GROUPS))
    out = tmp_path / "report.md"
    write_capture_report(store, STAGES[:2], items, GROUPS, out)
    text = out.read_text()
    assert "| S1 | gpt-3.5-turbo-0125 | 4 | 4 | 1.000 | 0.000 | 1.000 |" in text
    assert "| S2 | gpt-3.5-turbo-0125 | 4 | 0 |" in text
    assert "gpt-3.5-turbo-0125" in text and "fp_x" in text
