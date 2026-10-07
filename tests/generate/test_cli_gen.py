import yaml
from typer.testing import CliRunner

from tests.generate.conftest import FakeFrameCache, FakeRenderer
from tests.helpers.factories import make_item
from vjaudit.cli import app
from vjaudit.config import GeneratorsFile, load_yaml
from vjaudit.generate.coverage import run_subset
from vjaudit.generate.engine import FakeEngine
from vjaudit.store import ArtifactStore


def _store(tmp_path):
    store = ArtifactStore(tmp_path / "a" / "server")  # the CLI's default profile is server
    for i in range(4):
        it = make_item(i, item_id=f"nextoe:test:{i}", benchmark="nextoe", in_main=True, in_annotated=True,
                       mc_options=list("abcde"), mc_answer_idx=0)
        store.put("item", it, it.item_id)
    pf = tmp_path / "paths.yaml"
    pf.write_text(yaml.safe_dump({"artifacts": str(tmp_path / "a"), "frame_cache": str(tmp_path / "f")}))
    return store, pf


def test_gen_coverage_cli_reports_missing(tmp_path):
    _, pf = _store(tmp_path)
    res = CliRunner().invoke(app, ["gen", "coverage", "--subset", "annotated", "--paths", str(pf)])
    assert res.exit_code == 0, res.output
    assert "nextoe qwen3vl8b missing 4" in res.output and "TOTAL missing 44" in res.output


def test_gen_mc_report_writes_table(tmp_path):
    store, pf = _store(tmp_path)
    gens = load_yaml("configs/profiles/server/generators.yaml", GeneratorsFile).generators
    items = [__import__("vjaudit.schemas", fromlist=["Item"]).Item(**r) for r in store.iter("item")]
    run_subset(store, items, gens, "main", engine_factory=lambda g: FakeEngine(lambda r: "A"),
               renderer_factory=lambda g: FakeRenderer(), frame_cache=FakeFrameCache(), mc=True)
    out = tmp_path / "mc.md"
    res = CliRunner().invoke(app, ["gen", "mc-report", "--paths", str(pf), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "| qwen3vl8b-mc | 4 | 1.000 | 0 |" in out.read_text()
