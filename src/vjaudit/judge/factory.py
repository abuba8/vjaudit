"""Build the cached judge client for a judge config: OpenAI API or a local transformers model."""
from __future__ import annotations

from pathlib import Path

from vjaudit.config import JudgeCfg

from .api_openai import OpenAIJudgeClient
from .local import JudgeBackend, LocalTransport


def make_client(cfg: JudgeCfg, cache_dir: Path, backend: JudgeBackend | None = None,
                allow_unpinned: bool = False) -> OpenAIJudgeClient:
    """Local judges cache under <cache_dir>/<judge id>/<revision>, so a re-pinned model never reuses old answers."""
    if cfg.backend == "transformers":
        transport = LocalTransport(cfg, backend, allow_unpinned=allow_unpinned)
        return OpenAIJudgeClient(cfg, Path(cache_dir) / cfg.id / cfg.revision, transport=transport)
    return OpenAIJudgeClient(cfg, cache_dir)
