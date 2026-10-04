"""Yes/No logit margins from API top-logprobs (R-ARC-02).

margin = logsumexp(logprobs of yes-class tokens) − logsumexp(logprobs of no-class tokens) at the
verdict position. A token's class ignores surrounding whitespace, quotes and colons, and case.
"""
from __future__ import annotations

import math
import re
from typing import Literal

_STRIP = " \t\n'\":"
_PRED_KEY = re.compile(r"""pred['"]?\s*:\s*['"]?""")


def token_class(tok: str) -> Literal["yes", "no"] | None:
    s = tok.strip(_STRIP).lower()
    return s if s in ("yes", "no") else None  # type: ignore[return-value]


def _lse(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def margin_from_top(top: list[tuple[str, float]]) -> float | None:
    yes = [lp for t, lp in top if token_class(t) == "yes"]
    no = [lp for t, lp in top if token_class(t) == "no"]
    return _lse(yes) - _lse(no) if yes and no else None


def p_yes(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin))


def vcg_verdict_index(tokens: list[str]) -> int | None:
    """Index of the token holding the value of the ``pred`` key, or None if it is split/absent."""
    starts, text = [], ""
    for t in tokens:
        starts.append(len(text))
        text += t
    m = _PRED_KEY.search(text)
    if not m:
        return None
    for i, (s, t) in enumerate(zip(starts, tokens)):
        if s + len(t) > m.end():  # first token reaching past the key and opening quote
            return i if token_class(t) is not None else None
    return None


def _top(entry: dict) -> list[tuple[str, float]]:
    return [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]


def _readout_at(content: list[dict], idx: int | None) -> tuple[float | None, list[list]]:
    if idx is None or idx >= len(content):
        return None, []
    top = _top(content[idx])
    top5 = [[t, lp] for t, lp in sorted(top, key=lambda x: -x[1])[:5]]
    return margin_from_top(top), top5


def readout_vcg(content: list[dict]) -> tuple[float | None, list[list]]:
    return _readout_at(content, vcg_verdict_index([c["token"] for c in content]))


def readout_first_token(content: list[dict]) -> tuple[float | None, list[list]]:
    return _readout_at(content, 0 if content else None)
