"""Local open-model judges through transformers, behind the same transport contract as the OpenAI API.

``LocalTransport`` takes an OpenAI chat-completions request dict and returns an OpenAI-shaped response
(choices, message text, logprobs.content with top_logprobs, usage). So the cached client, capture, parsers,
readout and report run unchanged for local judges.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Protocol

import numpy as np

from vjaudit.config import JudgeCfg
from vjaudit.device import default_dtype, pick_device

from .api_openai import TransportError


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def content_from_logits(gen_ids: list[int], step_logits: list[np.ndarray], decode: Callable[[int], str],
                        k: int) -> list[dict]:
    """OpenAI-style ``logprobs.content`` from raw per-step logits: chosen token + top-k alternatives."""
    out = []
    for tid, logits in zip(gen_ids, step_logits):
        lp = _log_softmax(logits)
        top = np.argpartition(-lp, k - 1)[:k] if 0 < k < lp.size else np.arange(lp.size if k > 0 else 0)
        top = sorted(top.tolist(), key=lambda j: -lp[j])
        out.append({"token": decode(int(tid)), "logprob": float(lp[tid]),
                    "top_logprobs": [{"token": decode(j), "logprob": float(lp[j])} for j in top]})
    return out


class JudgeBackend(Protocol):
    def render(self, messages: list[dict]) -> str: ...

    def generate_once(self, prompt: str, max_new_tokens: int, temperature: float,
                      seed: int) -> tuple[list[int], list[np.ndarray], int]: ...

    def decode(self, token_id: int) -> str: ...

    def decode_all(self, ids: list[int]) -> str: ...


class HFCausalBackend:
    """A causal LM from the Hugging Face Hub at a pinned revision; loads on first use."""

    def __init__(self, cfg: JudgeCfg, device: str | None = None, allow_unpinned: bool = False):
        self.cfg, self.device, self.allow_unpinned = cfg, device, allow_unpinned
        self._tok = self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if self.cfg.revision == "UNPINNED" and not self.allow_unpinned:
            raise ValueError(f"{self.cfg.id}: revision is UNPINNED; run `make pin PROFILE=<profile>` first, "
                             "or pass --allow-unpinned for a quick test")
        import torch  # mac extra
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.device or pick_device(torch)
        rev = None if self.cfg.revision == "UNPINNED" else self.cfg.revision
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model, revision=rev,
                                                     dtype=getattr(torch, default_dtype(self.device)))
        self._model = model.to(self.device).eval()

    def render(self, messages: list[dict]) -> str:
        self._ensure()
        return self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             **self.cfg.chat_template_kwargs)

    def generate_once(self, prompt: str, max_new_tokens: int, temperature: float,
                      seed: int) -> tuple[list[int], list[np.ndarray], int]:
        import torch

        self._ensure()
        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        kw = dict(max_new_tokens=max_new_tokens, output_logits=True, return_dict_in_generate=True,
                  pad_token_id=self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id)
        if temperature == 0:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:  # plain sampling, like the API default (no top-p / top-k truncation)
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        with torch.no_grad():
            out = self._model.generate(**enc, **kw)
        n_prompt = enc["input_ids"].shape[1]
        ids = out.sequences[0][n_prompt:].tolist()
        logits = [step[0].float().cpu().numpy() for step in out.logits][:len(ids)]
        return ids, logits, n_prompt

    def decode(self, token_id: int) -> str:
        return self._tok.decode([token_id])

    def decode_all(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class LocalTransport:
    """Async callable with the OpenAI transport contract, backed by a local model (one call at a time)."""

    def __init__(self, cfg: JudgeCfg, backend: JudgeBackend | None = None, allow_unpinned: bool = False):
        self.cfg = cfg
        self.backend = backend or HFCausalBackend(cfg, allow_unpinned=allow_unpinned)

    async def __call__(self, req: dict) -> dict:
        try:
            return await asyncio.to_thread(self._run, req)
        except Exception as e:  # noqa: BLE001 — non-retriable: capture records it as failed and moves on
            raise TransportError(422, "local_error", f"{type(e).__name__}: {e}") from e

    def _run(self, req: dict) -> dict:
        be = self.backend
        prompt = be.render(req["messages"])
        temperature = 0.0 if req.get("temperature") == 0 else (req.get("temperature") or 1.0)
        choices, n_prompt, n_completion = [], 0, 0
        for k in range(req.get("n", 1)):
            ids, logits, n_prompt = be.generate_once(prompt, req["max_tokens"], temperature, seed=k)
            logprobs = ({"content": content_from_logits(ids, logits, be.decode, req.get("top_logprobs", 0))}
                        if req.get("logprobs") else None)
            choices.append({"index": k, "message": {"role": "assistant", "content": be.decode_all(ids)},
                            "logprobs": logprobs})
            n_completion += len(ids)
        return {"model": self.cfg.model, "system_fingerprint": self.cfg.revision, "choices": choices,
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion}}
