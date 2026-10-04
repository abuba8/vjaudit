"""Run one system over a list of items; one Generation per (item, system); resumable."""
from __future__ import annotations

import logging
from dataclasses import asdict

from vjaudit.config import GeneratorCfg
from vjaudit.data.frames import FrameDecodeError
from vjaudit.hashing import config_hash, short_id
from vjaudit.normalize import normalize_answer
from vjaudit.schemas import Generation, Item
from vjaudit.store import ArtifactStore, stamp

from .adapters import Adapter, load_prompt
from .engine import Engine
from .systems import SystemSpec

log = logging.getLogger(__name__)
MAX_NEW_TOKENS = {"short": 128, "sentence": 256, "mc": 8}


def gen_config(gen: GeneratorCfg, spec: SystemSpec) -> dict:
    """Everything that determines the output; its hash is part of gen_id."""
    return {"generator": gen.model_dump(), "spec": asdict(spec), "prompt": load_prompt(spec.style),
            "decoding": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS[spec.style]}}


def run_generation(store: ArtifactStore, items: list[Item], spec: SystemSpec, engine: Engine, frame_cache,
                   gen: GeneratorCfg, adapter: Adapter, batch_size: int = 16) -> int:
    cfg = gen_config(gen, spec)
    chash = config_hash(cfg)
    todo = [(it, short_id(it.item_id, spec.system_id, chash)) for it in items if it.available]
    todo = [(it, gid) for it, gid in todo if not store.has("generation", gid)]
    new = 0
    for b in range(0, len(todo), batch_size):
        batch = todo[b:b + batch_size]
        reqs, indices, kept = [], [], []
        for it, gid in batch:
            if spec.blind:
                frames, idx = None, []
            else:
                try:
                    frames, idx = frame_cache.get(it.video_path, it.video_id, spec.n_frames, it.t_start, it.t_end)
                except FrameDecodeError as e:  # skip this item, keep the run going; coverage shows the gap
                    log.warning("%s %s skipped: %s", spec.system_id, it.item_id, e)
                    continue
            reqs.append(adapter.build_request(it, frames, spec))
            indices.append(list(idx))
            kept.append((it, gid))
        if not reqs:
            continue
        batch = kept
        outs = engine.generate(reqs, cfg["decoding"]["max_new_tokens"])
        for (it, gid), idx, (text, n_tok) in zip(batch, indices, outs):
            rec = Generation(**stamp(dict(
                gen_id=gid, item_id=it.item_id, system_id=spec.system_id, generator_repo=gen.hf_repo,
                generator_rev=gen.revision, style=spec.style, blind=spec.blind, n_frames=0 if spec.blind else spec.n_frames,
                frame_indices=idx, decoding=cfg["decoding"], output_text=text, answer_norm=normalize_answer(text),
                n_output_tokens=n_tok), cfg))
            new += store.put("generation", rec, gid)
    return new
