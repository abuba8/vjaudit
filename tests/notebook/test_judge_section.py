"""Section 9: judging, pinned to the package's templates, parsers and margin readout."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.factories import make_item
from vjaudit.judge.local import content_from_logits
from vjaudit.judge.parse import parse_vcg, parse_yes_no
from vjaudit.judge.readout import readout_first_token, readout_vcg
from vjaudit.judge.templates import load_template, protocol_id, render_rub, render_vcg

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "configs" / "prompts"
FX = ROOT / "tests" / "fixtures" / "openai"


def test_render_and_protocol_match_package(nb_lib):
    q, a, p = "what is {this}?", "a {dog}", "{pred} cat"
    assert nb_lib["render"]("t-vcg", q, a, p, PROMPTS) == render_vcg(q, a, p)
    assert nb_lib["render"]("t-rub", q, a, p, PROMPTS) == render_rub(q, a, p)
    for tid in ("t-vcg", "t-rub"):
        assert nb_lib["protocol_id"](tid, "t0", PROMPTS) == protocol_id(load_template(tid), "t0")


@pytest.mark.parametrize("text", ["{'pred': 'yes', 'score': 4}", "```python\n{'pred': 'no', 'score': 1.5}\n```",
                                  "pred: Yes, score: 3", "I cannot judge", "{'pred': 'maybe'}", "", "{'pred': 'YES'}"])
def test_parse_vcg_matches_package(nb_lib, text):
    assert nb_lib["parse_vcg"](text) == parse_vcg(text)


@pytest.mark.parametrize("text", ["Yes", " no.", "YES!", "Yes, because", "", "Direct Match"])
def test_parse_yes_no_matches_package(nb_lib, text):
    assert nb_lib["parse_yes_no"](text) == parse_yes_no(text)


@pytest.mark.parametrize("name", sorted(p.name for p in FX.glob("*.json")))
def test_margin_matches_package(nb_lib, name):
    content = json.loads((FX / name).read_text())["content"]
    assert nb_lib["margin_at"](content, "t-vcg") == readout_vcg(content)
    assert nb_lib["margin_at"](content, "t-rub") == readout_first_token(content)


def test_content_from_logits_matches_package(nb_lib):
    vocab = [" No", " Yes", "Yes", "no", "{"]
    logits = [np.array([0.0, 1.0, 2.0, 0.5, -1.0]), np.array([3.0, 0.0, 0.0, 1.0, 9.0])]
    ours = nb_lib["content_from_logits"]([2, 4], logits, lambda i: vocab[i], 3)
    theirs = content_from_logits([2, 4], logits, lambda i: vocab[i], 3)
    assert ours == theirs


class FakeError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def _reply(text="Yes"):
    top = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.5}]
    return {"model": "gpt-3.5-turbo-0125", "system_fingerprint": "fp",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": {"content": [{"token": text, "logprob": -0.1, "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1}}


API_CFG = dict(id="gpt-3.5-turbo-0125", backend="openai", model="gpt-3.5-turbo-0125", max_usd=1.0,
               price_in_per_mtok=0.5, price_out_per_mtok=1.5)
REQ = dict(temperature=0, max_tokens=1, logprobs=True, top_logprobs=20, n=1)
MSGS = [{"role": "user", "content": "Is it right?"}]


def test_api_judge_retries_then_caches(nb_lib, tmp_path):
    calls = []

    def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise FakeError(429)
        return _reply()

    judge = nb_lib["APIJudge"](API_CFG, tmp_path, create=create, sleep=lambda s: None)
    r1 = judge.complete(MSGS, REQ)
    r2 = nb_lib["APIJudge"](API_CFG, tmp_path, create=create, sleep=lambda s: None).complete(MSGS, REQ)
    assert len(calls) == 2 and r1["choices"][0]["text"] == "Yes" and r2["cached"] is True
    assert calls[1]["temperature"] == 0 and calls[1]["top_logprobs"] == 20 and r1["cost"] > 0


def test_api_judge_retired_model_and_budget(nb_lib, tmp_path):
    def gone(**kw):
        raise FakeError(404)
    with pytest.raises(nb_lib["ModelRetired"]):
        nb_lib["APIJudge"](API_CFG, tmp_path, create=gone, sleep=lambda s: None).complete(MSGS, REQ)
    poor = dict(API_CFG, max_usd=0.0)
    with pytest.raises(nb_lib["BudgetExceeded"]):
        nb_lib["APIJudge"](poor, tmp_path / "b", create=lambda **kw: _reply(), sleep=lambda s: None).complete(MSGS, REQ)


def _judging_setup(nb_lib, tmp_path, prompts=PROMPTS):
    S = nb_lib["resolve_settings"](dict(PROFILE="mac", DRY_RUN=True, DATA_ROOT=str(tmp_path), RUN="annotated",
                                        RUN_LIMIT=None, RUN_STAGES=None), tmp_path / "repo")
    S["prompts"] = prompts
    store = nb_lib["Store"](S["runs"])
    items = pd.DataFrame([dict(make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True).model_dump(),
                               available=True, video_path="", problem="") for i in range(3)])
    store.append("items", items.to_dict("records"), key="item_id")
    gens = [dict(gen_id=f"g{i}{s}", item_id=f"msvd:test:{i}", benchmark="msvd", system_id=s, generator_id="x",
                 style="short", blind=s.endswith("blind"), output_text=["a dog", "a cat"][i % 2],
                 answer_norm=["dog", "cat"][i % 2]) for i in range(3) for s in ("qwen25vl3b", "qwen25vl3b-blind")]
    store.append("generations", gens, key="gen_id")
    return S, store, items


def test_run_stages_with_fake_judges(nb_lib, tmp_path):
    S, store, items = _judging_setup(nb_lib, tmp_path)
    summary = nb_lib["run_stages"](store, items, S)
    calls = store.load("judgecalls")
    assert list(summary["stage"]) == ["M1", "M2", "M3", "M4"] and summary["done"].sum() == 4 * 6
    assert (calls.parse_status == "ok").all() and calls.margin.notna().all()
    assert set(calls.columns) >= {"judge_call_id", "gen_id", "judge_id", "protocol_id", "parsed_verdict", "margin",
                                  "top_tokens", "raw_text", "parse_status", "sample_idx", "stage"}
    again = nb_lib["run_stages"](store, items, S)
    assert again["done"].sum() == 0 and again["jobs"].sum() == 0


def test_stage_failures_are_listed_not_fatal(nb_lib, tmp_path):
    S, store, items = _judging_setup(nb_lib, tmp_path)
    S["stages"] = S["stages"][:1]

    class Flaky(nb_lib["FakeJudge"]):
        def complete(self, messages, request):
            if "a cat" in json.dumps(messages):
                raise RuntimeError("judge crashed")
            return super().complete(messages, request)

    summary = nb_lib["run_stages"](store, items, S, judge_factory=lambda cfg, S: Flaky(cfg))
    assert summary.loc[0, "failed"] == 2 and summary.loc[0, "done"] == 4
    assert "judge crashed" in summary.loc[0, "errors"][0]


def test_template_change_gives_new_ids(nb_lib, tmp_path):
    copy = tmp_path / "prompts"
    shutil.copytree(PROMPTS, copy)
    S, store, items = _judging_setup(nb_lib, tmp_path)
    S["stages"] = S["stages"][1:2]  # M2, T-RUB
    nb_lib["run_stages"](store, items, S)
    S["prompts"] = copy
    (copy / "judge" / "t_rub.txt").write_text((copy / "judge" / "t_rub.txt").read_text() + "\nBe strict.")
    summary = nb_lib["run_stages"](store, items, S)
    assert summary.loc[0, "done"] == 6 and len(store.load("judgecalls")) == 12
