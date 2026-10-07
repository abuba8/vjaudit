"""Measure generation throughput (items/s) per pinned generator, video and blind (R-SP0-07). GPU only.

Usage: uv run python scripts/throughput.py --n 50 [--benchmark msvd]
Writes docs/throughput.md. Run one generator per GPU if memory is tight (--only <id>).
"""
import argparse
import time
from pathlib import Path

from vjaudit.data.frames import FrameCache
from vjaudit.generate.adapters import HFChatRenderer, get_adapter
from vjaudit.generate.engine import VllmEngine
from vjaudit.generate.systems import SystemSpec
from vjaudit.profile import load_profile
from vjaudit.schemas import Item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--benchmark", default="msvd")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="docs/throughput.md")
    a = ap.parse_args()
    prof = load_profile()  # $VJAUDIT_PROFILE, default server
    items = [Item(**r) for r in prof.store().iter("item")
             if r["benchmark"] == a.benchmark and r["in_annotated"] and r["available"]][: a.n]
    fc = FrameCache(prof.frame_cache)
    rows = []
    for g in prof.generators:
        if a.only and g.id != a.only:
            continue
        engine = VllmEngine(g)
        adapter = get_adapter(g.family, HFChatRenderer(g.hf_repo, g.revision, g.family))
        n_frames = g.frames_for(a.benchmark)
        for blind in (False, True):
            spec = SystemSpec("tp", g.id, "short", blind, 0 if blind else n_frames)
            reqs = [adapter.build_request(i, None if blind else fc.get(i.video_path, i.video_id, n_frames,
                                                                        i.t_start, i.t_end)[0], spec) for i in items]
            t0 = time.perf_counter()
            engine.generate(reqs, 128)
            dt = time.perf_counter() - t0
            rows.append((g.id, "blind" if blind else f"{n_frames} frames", len(reqs) / dt))
        del engine
    lines = ["# Generation throughput (A30, bf16, greedy, max_new_tokens 128)", "",
             "| generator | input | items/s |", "|---|---|---|"] + [f"| {g} | {m} | {r:.2f} |" for g, m, r in rows]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
