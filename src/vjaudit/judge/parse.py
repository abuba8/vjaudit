"""Parse judge outputs. Unparseable outputs are reported as such, never mapped to a verdict."""
from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
_PRED = re.compile(r"""pred['"]?\s*:\s*['"]?(yes|no)\b""", re.I)
_SCORE = re.compile(r"""score['"]?\s*:\s*['"]?([0-9]+(?:\.[0-9]+)?)""", re.I)


def _verdict(v) -> bool | None:
    s = str(v).strip().lower()
    return True if s == "yes" else False if s == "no" else None


def _score(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_vcg(text: str) -> tuple[bool | None, float | None, str]:
    """(verdict, score, status) for the Video-ChatGPT dict-string output."""
    t = _FENCE.sub("", text)
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            d = ast.literal_eval(t[a:b + 1])
            if isinstance(d, dict) and _verdict(d.get("pred")) is not None:
                return _verdict(d["pred"]), _score(d.get("score")), "ok"
        except (ValueError, SyntaxError):
            pass
    m = _PRED.search(t)
    if not m:
        return None, None, "unparseable"
    s = _SCORE.search(t)
    return m.group(1).lower() == "yes", float(s.group(1)) if s else None, "ok"


def parse_yes_no(text: str) -> tuple[bool | None, str]:
    s = text.strip().strip(".!,;:\"'").lower()
    if s == "yes":
        return True, "ok"
    if s == "no":
        return False, "ok"
    return None, "off_vocab"
