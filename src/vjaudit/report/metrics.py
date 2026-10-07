"""Small, pure agreement metrics on paired yes/no decisions (True = 'correct')."""
from __future__ import annotations


def yes_rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's κ = (p_o − p_e) / (1 − p_e); None when undefined (no pairs, or p_e = 1)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def error_rates(judge: list[bool], ref: list[bool]) -> dict:
    """FPR = judge says correct when the reference says incorrect; FNR = the reverse. None when undefined."""
    pos = [j for j, r in zip(judge, ref) if r]
    neg = [j for j, r in zip(judge, ref) if not r]
    return {"fpr": sum(neg) / len(neg) if neg else None,
            "fnr": sum(not j for j in pos) / len(pos) if pos else None,
            "n_ref_pos": len(pos), "n_ref_neg": len(neg)}
