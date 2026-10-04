"""Answer normalization for annotation-unit dedup (R-ANN-07)."""
from __future__ import annotations

import re
import string
import unicodedata

_PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")
_ARTICLE = re.compile(r"^(a|an|the) ")


def normalize_answer(text: str) -> str:
    """NFKC → lowercase → punctuation to spaces → collapse whitespace → strip one leading article.
    A non-empty input never maps to "" (falls back to the lowercased, stripped input)."""
    s = unicodedata.normalize("NFKC", text).lower()
    out = " ".join(_PUNCT.sub(" ", s).split())
    out = _ARTICLE.sub("", out, count=1)
    if not out and text.strip():
        return " ".join(s.split())
    return out
