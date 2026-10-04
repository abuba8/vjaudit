"""Cached, budgeted OpenAI chat-completions client for API judges (R-GLB-03, R-BUD-01, §7.1 #6).

* Every request is cached by sha256(canonical JSON of the request, model string included).
* 429 / 5xx / connection errors are retried with exponential backoff (2 s, ×2, jitter, max 6 attempts).
* ``model_not_found`` / HTTP 404 raises ``ModelRetired`` immediately (no retry).
* Before each non-cached call, ``spent_usd + estimate > max_usd`` raises ``BudgetExceeded``.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Awaitable, Callable

from vjaudit.config import JudgeCfg
from vjaudit.hashing import canonical_json, sha256_hex

MAX_ATTEMPTS = 6


class TransportError(Exception):
    def __init__(self, status: int, code: str | None, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code = status, code


class ModelRetired(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def request_hash(req: dict) -> str:
    return sha256_hex(canonical_json(req))


_ENC: list = []  # [encoder or None], resolved once


def _count_tokens(text: str) -> int:
    """tiktoken cl100k_base when available, else a conservative chars/3 estimate (resolved once)."""
    if not _ENC:
        try:
            import tiktoken

            _ENC.append(tiktoken.get_encoding("cl100k_base"))
        except Exception:  # noqa: BLE001 — no network / no cached BPE
            _ENC.append(None)
    return len(_ENC[0].encode(text)) if _ENC[0] is not None else len(text) // 3 + 1


def _retriable(e: TransportError) -> bool:
    return e.status in (0, 408, 409, 429) or e.status >= 500


def _is_retired(e: TransportError) -> bool:
    return e.code == "model_not_found" or e.status == 404


async def openai_transport(req: dict) -> dict:
    import openai

    client = openai.AsyncOpenAI(max_retries=0)
    try:
        resp = await client.chat.completions.create(**req)
    except openai.APIStatusError as e:
        body = e.body if isinstance(e.body, dict) else {}
        raise TransportError(e.status_code, body.get("code") or getattr(e, "code", None), str(e)) from e
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise TransportError(0, "connection", str(e)) from e
    return resp.model_dump()


class OpenAIJudgeClient:
    def __init__(self, cfg: JudgeCfg, cache_dir: Path, transport: Callable[[dict], Awaitable[dict]] | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.transport = transport or openai_transport
        self.sleep = sleep
        self.spent_usd = 0.0
        self._sem = asyncio.Semaphore(cfg.concurrency)

    def build_request(self, messages: list[dict], *, max_tokens: int, temperature: float | None, logprobs: bool,
                      top_logprobs: int, n: int = 1) -> dict:
        req = {"model": self.cfg.model, "messages": messages, "max_tokens": max_tokens, "n": n,
               "logprobs": logprobs, "top_logprobs": top_logprobs}
        if temperature is not None:
            req["temperature"] = temperature
        return req

    def estimate_cost(self, reqs: list[dict]) -> float:
        tin = sum(_count_tokens(m["content"]) + 4 for r in reqs for m in r["messages"])
        tout = sum(r["max_tokens"] * r.get("n", 1) for r in reqs)
        return (tin * self.cfg.price_in_per_mtok + tout * self.cfg.price_out_per_mtok) / 1e6

    def _cost(self, usage: dict) -> float:
        return (usage.get("prompt_tokens", 0) * self.cfg.price_in_per_mtok
                + usage.get("completion_tokens", 0) * self.cfg.price_out_per_mtok) / 1e6

    def _path(self, h: str) -> Path:
        return self.cache_dir / h[:2] / f"{h}.json"

    def cached(self, req: dict) -> dict | None:
        p = self._path(request_hash(req))
        if not p.exists():
            return None
        entry = json.loads(p.read_text(encoding="utf-8"))
        return {**entry["response"], "_cached": True, "_latency_s": entry["latency_s"], "_cost_usd": entry["cost_usd"]}

    async def complete(self, messages: list[dict], *, max_tokens: int, temperature: float | None, logprobs: bool,
                       top_logprobs: int, n: int = 1) -> dict:
        req = self.build_request(messages, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs,
                                 top_logprobs=top_logprobs, n=n)
        hit = self.cached(req)
        if hit is not None:
            return hit
        async with self._sem:
            if self.spent_usd + self.estimate_cost([req]) > self.cfg.max_usd:
                raise BudgetExceeded(f"{self.cfg.id}: spent {self.spent_usd:.4f} USD, cap {self.cfg.max_usd}")
            t0 = time.perf_counter()
            for attempt in range(MAX_ATTEMPTS):
                try:
                    resp = await self.transport(req)
                    break
                except TransportError as e:
                    if _is_retired(e):
                        raise ModelRetired(f"{self.cfg.model}: {e}") from e
                    if not _retriable(e) or attempt == MAX_ATTEMPTS - 1:
                        raise
                    await self.sleep(2.0 * 2 ** attempt * (1 + random.random() * 0.25))
            latency = time.perf_counter() - t0
            cost = self._cost(resp.get("usage") or {})
            self.spent_usd += cost
        p = self._path(request_hash(req))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": req, "response": resp, "latency_s": latency, "cost_usd": cost}),
                       encoding="utf-8")
        tmp.replace(p)
        return {**resp, "_cached": False, "_latency_s": latency, "_cost_usd": cost}
