"""Quick end-to-end check of each generator: a few items, with video and blind (run this first on the Mac)."""
from __future__ import annotations

import time
from typing import Callable

from vjaudit.schemas import Item

from .adapters import get_adapter
from .factory import make_engine, make_renderer
from .systems import SystemSpec


def smoke_generators(profile, items: list[Item], frame_cache, engine_factory: Callable = make_engine,
                     renderer_factory: Callable = make_renderer, n_items: int = 2) -> list[dict]:
    """One row per (generator, item, mode in video/blind). A generator that fails gets an 'ERROR: ...' row and
    the next generator still runs, so one bad model does not hide the others."""
    rows = []
    for g in profile.generators:
        engine = None
        try:
            engine = engine_factory(g)
            adapter = get_adapter(g.family, renderer_factory(g, engine))
            for it in items[:n_items]:
                for mode in ("video", "blind"):
                    blind = mode == "blind"
                    spec = SystemSpec("smoke", g.id, "short", blind, 0 if blind else g.frames_for(it.benchmark))
                    frames = None if blind else frame_cache.get(it.video_path, it.video_id, spec.n_frames,
                                                                it.t_start, it.t_end)[0]
                    t0 = time.perf_counter()
                    text, _ = engine.generate([adapter.build_request(it, frames, spec)], 128)[0]
                    rows.append({"generator": g.id, "item_id": it.item_id, "mode": mode, "answer": text,
                                 "seconds": round(time.perf_counter() - t0, 2),
                                 "device": getattr(engine, "device", "-")})
        except Exception as e:  # noqa: BLE001 — report and move on to the next generator
            rows.append({"generator": g.id, "item_id": "-", "mode": "-", "answer": f"ERROR: {e}", "seconds": 0.0,
                         "device": getattr(engine, "device", "-")})
        finally:
            if engine is not None and hasattr(engine, "close"):
                engine.close()
    return rows
