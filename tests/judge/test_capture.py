import asyncio
from pathlib import Path

import pytest

from tests.helpers.factories import make_generation, make_item
from vjaudit.config import GridCfg, JudgeCfg, load_yaml
from vjaudit.judge.api_openai import OpenAIJudgeClient, TransportError
from vjaudit.judge.capture import CaptureStage, build_jobs, load_stages, run_capture
from vjaudit.store import ArtifactStore

STAGES = load_stages(load_yaml("configs/profiles/server/grid.yaml", GridCfg))
GROUPS = {"core": {"qwen3vl8b", "qwen3vl8b-blind"}, "ablation": {"qwen3vl8b-f1"}}


def jc(model, mx=100.0):
    return JudgeCfg(id=model, kind="api", model=model, max_usd=mx, price_in_per_mtok=0.5, price_out_per_mtok=1.5,
                    concurrency=4)


def _choice(text, content):
    return {"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": {"content": content}}


def vcg_content(word="yes", split=False):
    toks = ["{'", "pred", "':", " '"] + (["y", "es"] if split else [word]) + ["',", " '", "score", "':", " 4", "}"]
    top = [{"token": "yes", "logprob": -0.2}, {"token": "no", "logprob": -1.9}]
    return [{"token": t, "logprob": -0.1, "top_logprobs": top if i == 4 else []} for i, t in enumerate(toks)]


def rub_content():
    return [{"token": "Yes", "logprob": -0.3, "top_logprobs": [{"token": "Yes", "logprob": -0.3}, {"token": "No", "logprob": -1.5}]}]


class Transport:
    def __init__(self, mode="ok"):
        self.calls, self.mode = [], mode

    async def __call__(self, req):
        self.calls.append(req)
        if self.mode == "retired":
            raise TransportError(404, "model_not_found", "gone")
        rub = req["max_tokens"] == 1
        if rub:
            choices = [_choice("Yes", rub_content())]
        elif self.mode == "garbage":
            choices = [_choice("I cannot judge this", vcg_content())]
        elif self.mode == "split":
            choices = [_choice("{'pred': 'yes', 'score': 4}", vcg_content(split=True))]
        else:
            choices = [_choice("{'pred': 'yes', 'score': 4}", vcg_content()) for _ in range(req["n"])]
        return {"model": req["model"], "system_fingerprint": "fp_x", "choices": choices,
                "usage": {"prompt_tokens": 300, "completion_tokens": 12 * req["n"]}}


def _store(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    items = [make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=i < 2) for i in range(3)]
    for it in items:
        store.put("item", it, it.item_id)
    gens = []
    for i in range(3):
        for k, sid in enumerate(["qwen3vl8b", "qwen3vl8b-blind", "qwen3vl8b-f1"]):
            g = make_generation(0, gen_id=f"g{sid[-2:]}{i}{k}", item_id=f"msvd:test:{i}", system_id=sid)
            store.put("generation", g, g.gen_id)
            gens.append(g)
    return store, items


def _clients(tmp_path, mode="ok", mode_1106=None):
    t0, t1 = Transport(mode), Transport(mode_1106 or mode)
    return {"gpt-3.5-turbo-0125": OpenAIJudgeClient(jc("gpt-3.5-turbo-0125"), tmp_path / "c", transport=t0),
            "gpt-3.5-turbo-1106": OpenAIJudgeClient(jc("gpt-3.5-turbo-1106"), tmp_path / "c", transport=t1)}, t0, t1


def test_stage_config_order_and_shape():
    assert [s.name for s in STAGES] == ["S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "S6d", "S7"]
    assert STAGES[4].request["n"] == 3 and STAGES[4].request["temperature"] is None
    assert STAGES[-1].judge_id == "gpt-3.5-turbo-1106"


def test_jobs_follow_stage_priority_then_gen_id(tmp_path):
    store, items = _store(tmp_path)
    jobs = build_jobs(store, STAGES, items, GROUPS)
    names = [j.stage.name for j in jobs]
    assert names == sorted(names, key=[s.name for s in STAGES].index)
    s1 = [j.gen["gen_id"] for j in jobs if j.stage.name == "S1"]
    assert s1 == sorted(s1) and len(s1) == 4  # 2 annotated items × 2 core systems
    s3 = [j for j in jobs if j.stage.name == "S3"]
    assert len(s3) == 2  # main = 3 items × 2 core systems, minus the 4 pairs S1 already schedules
    ids = [i for j in jobs for i in j.judge_call_ids]
    assert len(ids) == len(set(ids))


async def test_resume_skips_existing_judge_calls(tmp_path):
    store, items = _store(tmp_path)
    clients, t0, _ = _clients(tmp_path)
    s1 = [s for s in STAGES if s.name == "S1"]
    await run_capture(store, clients, build_jobs(store, s1, items, GROUPS))
    assert build_jobs(store, s1, items, GROUPS) == []
    s3 = [s for s in STAGES if s.name == "S3"]
    assert len(build_jobs(store, s3, items, GROUPS)) == 2  # S1 already covered the 4 annotated pairs


async def test_s5_writes_three_records_with_sample_idx(tmp_path):
    store, items = _store(tmp_path)
    clients, t0, _ = _clients(tmp_path)
    s5 = [s for s in STAGES if s.name == "S5"]
    summary = await run_capture(store, clients, build_jobs(store, s5, items, GROUPS))
    rows = [r for r in store.iter("judgecall") if r["protocol_id"].endswith("/default-n3")]
    assert len(rows) == 12 and sorted({r["sample_idx"] for r in rows}) == [0, 1, 2]
    assert all("temperature" not in c for c in t0.calls) and summary.done == 12


async def test_unparseable_is_recorded_with_status_not_dropped(tmp_path):
    store, items = _store(tmp_path)
    clients, _, _ = _clients(tmp_path, mode="garbage")
    await run_capture(store, clients, build_jobs(store, STAGES[:1], items, GROUPS))
    rows = list(store.iter("judgecall"))
    assert len(rows) == 4 and all(r["parse_status"] == "unparseable" and r["parsed_verdict"] is None for r in rows)


async def test_judgecall_fields(tmp_path):
    store, items = _store(tmp_path)
    clients, _, _ = _clients(tmp_path)
    await run_capture(store, clients, build_jobs(store, STAGES[:2], items, GROUPS))
    vcg = [r for r in store.iter("judgecall") if r["protocol_id"].startswith("t-vcg")]
    rub = [r for r in store.iter("judgecall") if r["protocol_id"].startswith("t-rub")]
    r = vcg[0]
    assert r["judge_rev"] == "gpt-3.5-turbo-0125" and r["api_model_returned"] == "gpt-3.5-turbo-0125"
    assert r["margin"] == pytest.approx(-0.2 - (-1.9)) and 0.5 < r["p_yes"] < 1 and len(r["top_tokens"]) <= 5
    assert r["vcg_score"] == 4.0 and r["parsed_verdict"] is True and r["parse_status"] == "ok"
    assert r["latency_s"] >= 0 and r["cost_usd"] > 0 and r["input_mode"] == "text" and r["demo_set_id"] is None
    assert r["system_fingerprint"] == "fp_x" and r["raw_text"].startswith("{'pred'")
    assert rub[0]["parsed_verdict"] is True and rub[0]["margin"] == pytest.approx(1.2) and rub[0]["vcg_score"] is None


async def test_no_logprob_keeps_parsed_verdict(tmp_path):
    store, items = _store(tmp_path)
    clients, _, _ = _clients(tmp_path, mode="split")
    await run_capture(store, clients, build_jobs(store, STAGES[:1], items, GROUPS))
    r = next(store.iter("judgecall"))
    assert r["margin"] is None and r["parse_status"] == "api_no_logprob" and r["parsed_verdict"] is True


async def test_model_retired_stops_and_writes_decision(tmp_path):
    store, items = _store(tmp_path)
    clients, _, _ = _clients(tmp_path, mode="retired")
    decisions = tmp_path / "decisions.md"
    summary = await run_capture(store, clients, build_jobs(store, STAGES[:1], items, GROUPS), decisions_path=decisions)
    assert summary.stopped_reason.startswith("ModelRetired") and summary.done == 0
    assert "gpt-3.5-turbo-0125 returned model_not_found at" in decisions.read_text()
    assert "captured 0 calls" in decisions.read_text()


async def test_s7_uses_1106_client(tmp_path):
    store, items = _store(tmp_path)
    clients, t0, t1 = _clients(tmp_path)
    await run_capture(store, clients, build_jobs(store, [s for s in STAGES if s.name == "S7"], items, GROUPS))
    assert t0.calls == [] and t1.calls and all(c["model"] == "gpt-3.5-turbo-1106" for c in t1.calls)
    rows = list(store.iter("judgecall"))
    assert len(rows) == 4 and all(r["judge_id"] == "gpt-3.5-turbo-1106" for r in rows)
    # identical requests (same question, same answer text) are served from the cache, not re-sent
    assert len(t1.calls) == 2


async def test_one_failing_request_is_recorded_and_capture_continues(tmp_path):
    store, items = _store(tmp_path)

    class Flaky(Transport):
        async def __call__(self, req):
            if "what is 0?" in req["messages"][-1]["content"]:
                self.calls.append(req)
                raise TransportError(400, "content_filter", "rejected")
            return await super().__call__(req)

    t = Flaky()
    clients = {"gpt-3.5-turbo-0125": OpenAIJudgeClient(jc("gpt-3.5-turbo-0125"), tmp_path / "c", transport=t)}
    summary = await run_capture(store, clients, build_jobs(store, STAGES[:1], items, GROUPS))
    assert summary.stopped_reason == "" and summary.failed == 2 and summary.done == 2
    assert any("content_filter" in e for e in summary.errors)
