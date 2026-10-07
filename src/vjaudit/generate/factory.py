"""Choose the generation engine and chat-template renderer from a generator's backend."""
from __future__ import annotations

from vjaudit.config import GeneratorCfg

from .adapters import HFChatRenderer, Renderer
from .engine import Engine, VllmEngine
from .hf_engine import TransformersEngine


def make_engine(gen: GeneratorCfg, allow_unpinned: bool = False) -> Engine:
    """backend 'vllm' → VllmEngine (GPU server); 'hf' → TransformersEngine (Mac / any torch device)."""
    if gen.backend == "hf":
        return TransformersEngine(gen, allow_unpinned=allow_unpinned)
    return VllmEngine(gen, allow_unpinned=allow_unpinned)


def make_renderer(gen: GeneratorCfg, engine) -> Renderer:
    """A TransformersEngine renders with its own processor; vLLM engines use the pinned HF chat template."""
    if isinstance(engine, TransformersEngine):
        return engine.render
    return HFChatRenderer(gen.hf_repo, gen.revision, gen.family)
