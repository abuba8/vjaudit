"""Systems = generator × answer style (× blind / frame count), plus the NExT MC-anchor systems."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vjaudit.config import GeneratorCfg


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    generator_id: str
    style: Literal["short", "sentence", "mc"]
    blind: bool
    n_frames: int


def expand_systems(gens: list[GeneratorCfg], benchmark: str) -> list[SystemSpec]:
    """Judged systems: <gen> (short), <gen>-sentence, <gen>-blind, <gen>-f<k>."""
    out: list[SystemSpec] = []
    for g in gens:
        n = g.frames_for(benchmark)
        for style in g.styles:
            out.append(SystemSpec(g.id if style == "short" else f"{g.id}-{style}", g.id, style, False, n))
        if g.blind:
            out.append(SystemSpec(f"{g.id}-blind", g.id, "short", True, 0))
        for k in g.frame_ablation:
            out.append(SystemSpec(f"{g.id}-f{k}", g.id, "short", False, k))
    return out


def expand_mc_systems(gens: list[GeneratorCfg]) -> list[SystemSpec]:
    """<gen>-mc for every generator; NExT only (judge-free anchor, G-ANCHOR)."""
    return [SystemSpec(f"{g.id}-mc", g.id, "mc", False, g.frames_for("nextoe")) for g in gens]


_LETTER = re.compile(r"\b([A-E])\b")


def parse_mc_letter(text: str) -> int | None:
    """Leading A–E not followed by a letter, else the first standalone A–E; A..E → 0..4."""
    s = text.strip()
    if s and s[0] in "ABCDE" and (len(s) == 1 or not s[1].isalpha()):
        return "ABCDE".index(s[0])
    m = _LETTER.search(s)
    return "ABCDE".index(m.group(1)) if m else None
