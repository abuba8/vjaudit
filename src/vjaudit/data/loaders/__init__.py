"""Benchmark loaders → list[Item]."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from vjaudit.schemas import Item

from .anetqa import load_anetqa
from .msvd import load_msvd, qtype_msvd
from .nextqa import load_nextoe

LOADERS = {"msvd": load_msvd, "anetqa": load_anetqa, "nextoe": load_nextoe}


def load_benchmark(name: Literal["msvd", "anetqa", "nextoe"], raw_dir: Path, split: str = "test") -> list[Item]:
    return LOADERS[name](Path(raw_dir), split)


__all__ = ["load_benchmark", "qtype_msvd", "LOADERS"]
