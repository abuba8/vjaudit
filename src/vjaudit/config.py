"""Pydantic models for every YAML file under configs/ (configuration as data, R-GLB-06)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T", bound=BaseModel)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unique_ids(items: list, what: str) -> None:
    ids = [x.id for x in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate {what} ids: {sorted(dupes)}")


class BenchmarkCfg(_Strict):
    id: Literal["msvd", "anetqa", "nextoe"]
    split: str = "test"
    raw_dir_key: str  # key into configs/local/paths.yaml
    video_dir_key: str  # key into configs/local/paths.yaml
    files: dict[str, str]
    video_exts: list[str] = [".mp4", ".avi", ".mkv", ".webm"]


class BenchmarksFile(_Strict):
    benchmarks: list[BenchmarkCfg]

    @model_validator(mode="after")
    def _ids(self):
        _unique_ids(self.benchmarks, "benchmark")
        return self


class GeneratorCfg(_Strict):
    id: str
    family: Literal["qwen-vl", "internvl", "llava-next-video", "llava-onevision"]
    hf_repo: str
    revision: str
    backend: Literal["vllm", "hf"]
    dtype: str = "bfloat16"
    n_frames: dict[str, int]  # benchmark id -> frames; "default" fallback
    max_model_len: int
    styles: list[Literal["short", "sentence"]]
    blind: bool
    frame_ablation: list[int]
    quantization: str | None = None

    def frames_for(self, benchmark: str) -> int:
        return self.n_frames.get(benchmark, self.n_frames["default"])


class GeneratorsFile(_Strict):
    generators: list[GeneratorCfg]

    @model_validator(mode="after")
    def _ids(self):
        _unique_ids(self.generators, "generator")
        return self


class JudgeCfg(_Strict):
    id: str
    kind: Literal["api", "open"]
    model: str
    max_usd: float
    price_in_per_mtok: float
    price_out_per_mtok: float
    concurrency: int
    backend: Literal["openai", "transformers"] = "openai"  # transformers = local open model (Mac profile)
    revision: str = "UNPINNED"  # HF revision SHA for local judges; unused for API judges
    chat_template_kwargs: dict = {}


class JudgesFile(_Strict):
    judges: list[JudgeCfg]

    @model_validator(mode="after")
    def _ids(self):
        _unique_ids(self.judges, "judge")
        return self


class GridCfg(_Strict):
    name: str
    seed: int
    benchmarks: list[str]
    sample_sizes: dict[str, int]
    msvd_min_per: int
    frac_dev: float
    frame_max_side: int = 768  # longest frame side after resizing (frame cache)
    capture_stages: list[dict] = []


def load_yaml(path: Path | str, model: type[T]) -> T:
    with open(path, encoding="utf-8") as f:
        return model.model_validate(yaml.safe_load(f))
