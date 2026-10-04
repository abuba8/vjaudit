import asyncio
import json

import pytest

from vjaudit.config import JudgeCfg
from vjaudit.judge.api_openai import BudgetExceeded, ModelRetired, OpenAIJudgeClient, TransportError, request_hash

CFG = JudgeCfg(id="gpt-3.5-turbo-0125", kind="api", model="gpt-3.5-turbo-0125", max_usd=1.0,
               price_in_per_mtok=0.5, price_out_per_mtok=1.5, concurrency=4)
MSGS = [{"role": "user", "content": "Is it a dog? Answer yes or no."}]


def ok_response(text="yes", pt=100, ct=10):
    return {"id": "x", "model": "gpt-3.5-turbo-0125", "system_fingerprint": None,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": {"content": [{"token": text, "logprob": -0.1, "top_logprobs": []}]}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


class FakeTransport:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, req):
        self.calls.append(req)
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


async def _no_sleep(_):
    return None


def client(tmp_path, transport, cfg=CFG):
    return OpenAIJudgeClient(cfg, tmp_path / "cache", transport=transport, sleep=_no_sleep)


async def test_cache_hit_makes_no_call(tmp_path):
    t = FakeTransport([ok_response()])
    c = client(tmp_path, t)
    r1 = await c.complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    r2 = await client(tmp_path, t).complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    assert len(t.calls) == 1 and r1["_cached"] is False and r2["_cached"] is True
    assert r2["choices"][0]["message"]["content"] == "yes" and r2["_cost_usd"] == r1["_cost_usd"]


async def test_retries_429_then_succeeds(tmp_path):
    t = FakeTransport([TransportError(429, "rate_limit_exceeded", "slow down"),
                       TransportError(500, "server_error", "oops"), ok_response()])
    r = await client(tmp_path, t).complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    assert len(t.calls) == 3 and r["choices"][0]["message"]["content"] == "yes"


async def test_gives_up_after_6_attempts(tmp_path):
    t = FakeTransport([TransportError(429, "rate_limit_exceeded", "x")] * 6)
    with pytest.raises(TransportError):
        await client(tmp_path, t).complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    assert len(t.calls) == 6


async def test_model_not_found_raises_model_retired_without_retry(tmp_path):
    t = FakeTransport([TransportError(404, "model_not_found", "The model does not exist")])
    with pytest.raises(ModelRetired):
        await client(tmp_path, t).complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    assert len(t.calls) == 1


async def test_budget_exceeded_before_sending(tmp_path):
    tiny = CFG.model_copy(update={"max_usd": 1e-9})
    t = FakeTransport([ok_response()])
    with pytest.raises(BudgetExceeded):
        await client(tmp_path, t, tiny).complete(MSGS, max_tokens=64, temperature=0, logprobs=True, top_logprobs=20)
    assert t.calls == []


def test_request_hash_stable_across_dict_order():
    a = {"model": "m", "messages": MSGS, "max_tokens": 1}
    b = {"max_tokens": 1, "messages": MSGS, "model": "m"}
    assert request_hash(a) == request_hash(b)


async def test_spent_usd_uses_usage_not_estimate(tmp_path):
    c = client(tmp_path, FakeTransport([ok_response(pt=1000, ct=100)]))
    await c.complete(MSGS, max_tokens=64, temperature=0, logprobs=True, top_logprobs=20)
    assert c.spent_usd == pytest.approx(1000 * 0.5 / 1e6 + 100 * 1.5 / 1e6)


async def test_temperature_none_is_omitted_from_request(tmp_path):
    t = FakeTransport([ok_response()])
    await client(tmp_path, t).complete(MSGS, max_tokens=64, temperature=None, logprobs=True, top_logprobs=20, n=3)
    assert "temperature" not in t.calls[0] and t.calls[0]["n"] == 3 and t.calls[0]["model"] == "gpt-3.5-turbo-0125"


async def test_concurrency_is_bounded(tmp_path):
    live = {"now": 0, "max": 0}

    async def slow(req):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.01)
        live["now"] -= 1
        return ok_response()

    c = client(tmp_path, slow, CFG.model_copy(update={"concurrency": 2}))
    await asyncio.gather(*[c.complete([{"role": "user", "content": f"q{i}"}], max_tokens=1, temperature=0,
                                      logprobs=True, top_logprobs=20) for i in range(8)])
    assert live["max"] == 2


@pytest.mark.api
async def test_live_smoke(tmp_path):
    """Real call (server only): T-VCG with logprobs returns token logprobs and the pinned snapshot."""
    from vjaudit.judge.templates import render_vcg
    c = OpenAIJudgeClient(CFG.model_copy(update={"max_usd": 0.05}), tmp_path / "cache")
    r = await c.complete(render_vcg("what is the man doing?", "cooking", "he is cooking"), max_tokens=64,
                         temperature=0, logprobs=True, top_logprobs=20)
    assert r["choices"][0]["logprobs"]["content"] and r["model"].startswith("gpt-3.5-turbo-0125")
