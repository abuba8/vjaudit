import pytest
import yaml
from typer.testing import CliRunner

from tests.helpers.factories import make_generation, make_item
from tests.judge.test_local import CFG, VCG_SCRIPT, FakeBackend
from vjaudit.cli import app
from vjaudit.cli_judge import _groups
from vjaudit.config import GridCfg, load_yaml
from vjaudit.judge.capture import build_jobs, load_stages, run_capture, select_stages
from vjaudit.judge.factory import make_client
from vjaudit.store import ArtifactStore

MAC = load_stages(load_yaml("configs/profiles/mac/grid.yaml", GridCfg))
MAC_SYSTEMS = {"qwen25vl3b", "qwen25vl3b-sentence", "qwen25vl3b-blind", "qwen3vl2b", "llavaov05b"}


class JudgeLikeBackend(FakeBackend):
    """Answers T-VCG (64 tokens) with a dict string and T-RUB (1 token) with 'Yes'."""

    def generate_once(self, prompt, max_new_tokens, temperature, seed):
        self.script = list(VCG_SCRIPT) if max_new_tokens > 1 else ["Yes"]
        return super().generate_once(prompt, max_new_tokens, temperature, seed)


def test_mac_stage_order_and_optional():
    assert [s.name for s in MAC] == ["M1", "M2", "M3", "M4", "M5", "M6", "G1", "G2"]
    assert [s.optional for s in MAC] == [False] * 6 + [True, True]
    assert [s.name for s in select_stages(MAC, [])] == ["M1", "M2", "M3", "M4", "M5", "M6"]


def test_select_named_optional_stage():
    assert [s.name for s in select_stages(MAC, ["G1"])] == ["G1"]
    assert [s.name for s in select_stages(MAC, ["M2", "M1"])] == ["M1", "M2"]  # config order


def test_select_unknown_stage_names_the_choices():
    with pytest.raises(ValueError, match="M7.*M1"):
        select_stages(MAC, ["M7"])


def _store(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    items = [make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True) for i in range(2)]
    for it in items:
        store.put("item", it, it.item_id)
        for sid in ("qwen25vl3b", "qwen25vl3b-blind"):
            g = make_generation(0, gen_id=f"{sid}-{it.item_id}", item_id=it.item_id, system_id=sid)
            store.put("generation", g, g.gen_id)
    return store, items


async def test_capture_with_local_judge_end_to_end(tmp_path):
    store, items = _store(tmp_path)
    be = JudgeLikeBackend()
    clients = {"qwen3-4b": make_client(CFG, tmp_path / "cache", backend=be)}
    groups = {"core": {"qwen25vl3b", "qwen25vl3b-blind"}, "ablation": set()}
    summary = await run_capture(store, clients, build_jobs(store, MAC[:2], items, groups))
    calls = list(store.iter("judgecall"))
    assert summary.done == 8 and summary.failed == 0 and len(calls) == 8
    assert all(c["parse_status"] == "ok" and c["margin"] is not None for c in calls)
    assert all(c["judge_rev"] == CFG.revision and c["cost_usd"] == 0 for c in calls)
    assert {c["parsed_verdict"] for c in calls} == {True}


async def test_local_judge_error_is_recorded_and_capture_continues(tmp_path):
    store, items = _store(tmp_path)
    clients = {"qwen3-4b": make_client(CFG, tmp_path / "cache", backend=FakeBackend(fail=RuntimeError("oom")))}
    groups = {"core": {"qwen25vl3b", "qwen25vl3b-blind"}, "ablation": set()}
    summary = await run_capture(store, clients, build_jobs(store, MAC[:1], items, groups))
    assert summary.failed == 4 and summary.done == 0 and "local_error" in summary.errors[0]


def test_groups_without_ablation_is_all_systems():
    gens = load_yaml("configs/profiles/mac/generators.yaml",
                     __import__("vjaudit.config", fromlist=["GeneratorsFile"]).GeneratorsFile).generators
    g = _groups(gens)
    assert g["core"] == MAC_SYSTEMS and g["ablation"] == set()


def test_capture_dry_run_mac_profile(tmp_paths):
    store = ArtifactStore(tmp_paths.parent / "a" / "mac")
    it = make_item(0, in_main=True, in_annotated=True)
    store.put("item", it, it.item_id)
    g = make_generation(0, gen_id="g0", item_id=it.item_id, system_id="qwen3vl2b")
    store.put("generation", g, g.gen_id)
    res = CliRunner().invoke(app, ["judge", "capture", "--profile", "mac", "--paths", str(tmp_paths),
                                   "--allow-unverified"])
    assert res.exit_code == 0, res.output
    assert "M1: 1 jobs, estimated ≤ 0.00 USD (qwen3-4b" in res.output and "M4: 1 jobs" in res.output
    assert "M5: 0 jobs" in res.output  # same judge call as M1 for an item in both sets: judged once
    assert "G1" not in res.output and "dry run" in res.output
