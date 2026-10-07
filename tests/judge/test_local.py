import math
from types import SimpleNamespace

import numpy as np
import pytest

from vjaudit.config import JudgeCfg
from vjaudit.judge.api_openai import TransportError
from vjaudit.judge.factory import make_client
from vjaudit.judge.local import HFCausalBackend, LocalTransport, content_from_logits
from vjaudit.judge.readout import readout_first_token, readout_vcg
from vjaudit.judge.smoke import smoke_judges

CFG = JudgeCfg(id="qwen3-4b", kind="open", backend="transformers", model="Qwen/Qwen3-4B-Instruct-2507",
               revision="rev1", max_usd=0, price_in_per_mtok=0, price_out_per_mtok=0, concurrency=1)
API = JudgeCfg(id="gpt-3.5-turbo-0125", kind="api", model="gpt-3.5-turbo-0125", max_usd=1,
               price_in_per_mtok=0.5, price_out_per_mtok=1.5, concurrency=1)
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


class FakeBackend:
    """A tiny 'model': emits a scripted token sequence; the chosen token gets logit 5, its yes/no rival 3."""

    VOCAB = ["Yes", " Yes", "No", " No", "{'", "pred", "':", " '", "yes", "no", "',", " 'score", " 4", "}", "<end>"]

    def __init__(self, script=("Yes",), fail=None):
        self.script, self.fail, self.calls = list(script), fail, []

    def render(self, messages):
        return "|".join(m["content"] for m in messages)

    def generate_once(self, prompt, max_new_tokens, temperature, seed):
        if self.fail:
            raise self.fail
        self.calls.append((temperature, seed))
        ids, logits = [], []
        for tok in self.script[:max_new_tokens]:
            i = self.VOCAB.index(tok)
            row = np.zeros(len(self.VOCAB))
            row[i] = 5.0
            rival = {"Yes": "No", "No": "Yes", "yes": "no", "no": "yes"}.get(tok)
            if rival:
                row[self.VOCAB.index(rival)] = 3.0
            ids.append(i)
            logits.append(row)
        return ids, logits, 7

    def decode(self, token_id):
        return self.VOCAB[token_id]

    def decode_all(self, ids):
        return "".join(self.VOCAB[i] for i in ids if self.VOCAB[i] != "<end>")


# The chosen yes/no token has logit 5, its rival 3, and the other two spellings of each class 0; the margin
# sums every spelling of a class: log(e^5 + 2) - log(e^3 + 2).
FAKE_MARGIN = math.log(math.exp(5) + 2) - math.log(math.exp(3) + 2)
VCG_SCRIPT = ("{'", "pred", "':", " '", "yes", "',", " 'score", "':", " 4", "}", "<end>")


def test_content_from_logits_values():
    logits = [np.array([0.0, 1.0, 2.0]), np.array([3.0, 0.0, 0.0])]
    out = content_from_logits([2, 0], logits, decode=lambda i: [" No", " Yes", "Yes"][i], k=2)
    lse0 = math.log(math.exp(0) + math.exp(1) + math.exp(2))
    assert out[0]["token"] == "Yes" and out[0]["logprob"] == pytest.approx(2 - lse0)
    assert [t["token"] for t in out[0]["top_logprobs"]] == ["Yes", " Yes"]
    assert out[0]["top_logprobs"][1]["logprob"] == pytest.approx(1 - lse0)
    assert out[1]["token"] == " No" and out[1]["top_logprobs"][0]["token"] == " No"
    assert all(isinstance(e["logprob"], float) for e in out)


def test_content_from_logits_is_stable_for_large_logits():
    out = content_from_logits([0], [np.array([1000.0, 999.0])], decode=str, k=2)
    assert out[0]["logprob"] == pytest.approx(-math.log(1 + math.exp(-1)))


def test_margin_from_local_content_aggregates_space_variants():
    vocab = [" No", " Yes", "Yes", "no"]
    content = content_from_logits([2], [np.array([0.0, 1.0, 2.0, 0.5])], decode=lambda i: vocab[i], k=4)
    margin, top = readout_first_token(content)
    assert margin == pytest.approx(math.log(math.e ** 1 + math.e ** 2) - math.log(1 + math.e ** 0.5))
    assert top[0][0] == "Yes"


async def test_local_transport_shapes_response():
    r = await LocalTransport(CFG, FakeBackend())({"model": "m", "messages": MSGS, "max_tokens": 1, "n": 1,
                                                  "logprobs": True, "top_logprobs": 2, "temperature": 0})
    assert r["model"] == CFG.model and r["system_fingerprint"] == "rev1"
    ch = r["choices"][0]
    assert ch["index"] == 0 and ch["message"] == {"role": "assistant", "content": "Yes"}
    assert r["usage"] == {"prompt_tokens": 7, "completion_tokens": 1}
    assert len(ch["logprobs"]["content"][0]["top_logprobs"]) == 2


async def test_local_transport_vcg_margin_readable():
    r = await LocalTransport(CFG, FakeBackend(VCG_SCRIPT))({"model": "m", "messages": MSGS, "max_tokens": 64,
                                                            "n": 1, "logprobs": True, "top_logprobs": 20,
                                                            "temperature": 0})
    assert r["choices"][0]["message"]["content"] == "{'pred': 'yes', 'score': 4}"
    margin, _ = readout_vcg(r["choices"][0]["logprobs"]["content"])
    assert margin == pytest.approx(FAKE_MARGIN)


async def test_local_transport_n3_uses_sampling_seeds():
    be = FakeBackend()
    r = await LocalTransport(CFG, be)({"model": "m", "messages": MSGS, "max_tokens": 1, "n": 3, "logprobs": True,
                                       "top_logprobs": 2})
    assert be.calls == [(1.0, 0), (1.0, 1), (1.0, 2)] and len(r["choices"]) == 3
    assert [c["index"] for c in r["choices"]] == [0, 1, 2] and r["usage"]["completion_tokens"] == 3


async def test_local_transport_error_is_422():
    with pytest.raises(TransportError) as ei:
        await LocalTransport(CFG, FakeBackend(fail=RuntimeError("boom")))({"model": "m", "messages": MSGS,
                                                                          "max_tokens": 1, "n": 1,
                                                                          "logprobs": True, "top_logprobs": 2,
                                                                          "temperature": 0})
    assert ei.value.status == 422 and ei.value.code == "local_error" and "boom" in str(ei.value)


async def test_client_with_local_transport_caches(tmp_path):
    be = FakeBackend()
    for _ in range(2):
        c = make_client(CFG, tmp_path, backend=be)
        r = await c.complete(MSGS, max_tokens=1, temperature=0, logprobs=True, top_logprobs=2)
    assert len(be.calls) == 1 and r["_cached"] is True and r["_cost_usd"] == 0


async def test_cache_is_separate_per_revision(tmp_path):
    be = FakeBackend()
    await make_client(CFG, tmp_path, backend=be).complete(MSGS, max_tokens=1, temperature=0, logprobs=True,
                                                          top_logprobs=2)
    other = CFG.model_copy(update={"revision": "rev2"})
    await make_client(other, tmp_path, backend=be).complete(MSGS, max_tokens=1, temperature=0, logprobs=True,
                                                            top_logprobs=2)
    assert len(be.calls) == 2


def test_make_client_for_api_judge_uses_default_transport(tmp_path):
    from vjaudit.judge.api_openai import openai_transport
    assert make_client(API, tmp_path).transport is openai_transport


def test_smoke_judges_rows():
    prof = SimpleNamespace(judges=[CFG, API])
    rows = smoke_judges(prof, client_factory=lambda cfg, d: make_client(
        cfg, d, backend=FakeBackend()))
    assert len(rows) == 2 and [r["template"] for r in rows] == ["t-vcg", "t-rub"]
    assert all(set(r) == {"judge", "template", "verdict", "margin", "top_tokens", "seconds"} for r in rows)
    assert rows[1]["verdict"] is True and rows[1]["margin"] == pytest.approx(FAKE_MARGIN)


def test_smoke_judges_reports_errors():
    prof = SimpleNamespace(judges=[CFG])
    rows = smoke_judges(prof, client_factory=lambda cfg, d: make_client(cfg, d, backend=FakeBackend(
        fail=RuntimeError("no weights"))))
    assert all(str(r["verdict"]).startswith("ERROR:") and "no weights" in r["verdict"] for r in rows)


def test_hf_backend_refuses_unpinned():
    be = HFCausalBackend(CFG.model_copy(update={"revision": "UNPINNED"}))
    with pytest.raises(ValueError, match="UNPINNED"):
        be.render(MSGS)


def test_judge_generate_kwargs_greedy_and_sampling():
    from vjaudit.judge.local import generate_kwargs
    g = generate_kwargs(0.0, 64, pad_token_id=5)
    assert g == {"max_new_tokens": 64, "output_logits": True, "return_dict_in_generate": True, "pad_token_id": 5,
                 "repetition_penalty": 1.0, "do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    s = generate_kwargs(1.0, 1, pad_token_id=5)
    assert s["do_sample"] is True and s["temperature"] == 1.0 and s["top_p"] == 1.0 and s["top_k"] == 0
    assert s["repetition_penalty"] == 1.0


async def test_local_judges_never_run_at_the_same_time():
    """Two local judges (e.g. at the M2→M3 boundary) must not run models concurrently on one GPU."""
    import asyncio
    import threading
    import time

    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    class SlowBackend(FakeBackend):
        def generate_once(self, prompt, max_new_tokens, temperature, seed):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return super().generate_once(prompt, max_new_tokens, temperature, seed)

    req = {"model": "m", "messages": MSGS, "max_tokens": 1, "n": 1, "logprobs": True, "top_logprobs": 2,
           "temperature": 0}
    a, b = LocalTransport(CFG, SlowBackend()), LocalTransport(API.model_copy(update={"id": "j2"}), SlowBackend())
    await asyncio.gather(a(req), b(req), a(req), b(req))
    assert state["max"] == 1
