"""Generation engines: vLLM on the GPU server, a fake one for tests."""
from __future__ import annotations

from typing import Callable, Protocol

from vjaudit.config import GeneratorCfg


class Engine(Protocol):
    def generate(self, requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]: ...


class FakeEngine:
    def __init__(self, fn: Callable[[dict], str]):
        self.fn = fn

    def generate(self, requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]:
        out = []
        for r in requests:
            text = self.fn(r)
            out.append((text, len(text.split())))
        return out


class VllmEngine:
    """Greedy decoding (temperature 0) with vLLM; refuses an unpinned revision unless allowed."""

    def __init__(self, gen: GeneratorCfg, allow_unpinned: bool = False, **llm_kwargs):
        from vllm import LLM  # gpu extra

        if gen.revision == "UNPINNED" and not allow_unpinned:
            raise ValueError(f"{gen.id}: revision is UNPINNED; run `vjaudit.cli gen pin-revisions` first (R-GLB-03)")
        self.llm = LLM(
            model=gen.hf_repo, revision=None if gen.revision == "UNPINNED" else gen.revision, dtype=gen.dtype,
            max_model_len=gen.max_model_len, limit_mm_per_prompt={"video": 1}, seed=0,
            trust_remote_code=gen.family == "internvl", enable_prefix_caching=False,
            quantization=gen.quantization, **llm_kwargs,
        )

    def generate(self, requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = self.llm.generate(requests, params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]
