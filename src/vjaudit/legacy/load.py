"""Load the prior (BlackboxNLP) pipeline's cached outputs, described by configs/local/legacy.yaml."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LegacyCell:
    dataset: str  # "nextqa" | "msvd"
    split: str  # "val" | "test"
    frames: int  # 4 | 8
    condition: str  # "zero" | "one" | "few"
    set: str | None  # "S1".."S5"; None for zero-shot
    gen_path: Path
    judge_path: Path

    def gen(self) -> dict:
        return json.loads(Path(self.gen_path).read_text(encoding="utf-8"))

    def judge(self) -> dict:
        return json.loads(Path(self.judge_path).read_text(encoding="utf-8"))


def load_legacy_config(cfg_path: Path | str) -> dict:
    return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))


def load_legacy_cells(cfg_path: Path | str) -> list[LegacyCell]:
    cfg = load_legacy_config(cfg_path)
    return [
        LegacyCell(dataset=c["dataset"], split=c["split"], frames=int(c["frames"]),
                   condition=c["condition"], set=c.get("set"),
                   gen_path=Path(c["gen_path"]), judge_path=Path(c["judge_path"]))
        for c in cfg["cells"]
    ]


def _yes_no_values(obj) -> list[str]:
    """All string values equal to Yes/No (any case) found recursively in ``obj``."""
    if isinstance(obj, str):
        return [obj.strip()] if obj.strip().lower() in {"yes", "no"} else []
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _yes_no_values(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _yes_no_values(x)]
    return []


def s1_one_shot_labels(prompts_dir: Path) -> list[str]:
    """Verdict label(s) in the first one-shot exemplar set (S1, called K1 in the prior repo)."""
    data = json.loads((Path(prompts_dir) / "examples" / "one_shot.json").read_text(encoding="utf-8"))
    if isinstance(data, dict):
        key = next((k for k in data if k.strip().upper() in {"K1", "S1"}), next(iter(data)))
        first = data[key]
    else:
        first = data[0]
    return _yes_no_values(first)
