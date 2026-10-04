"""Resolve each item's video file and check it decodes; write docs/data_availability.md."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import av

from vjaudit.schemas import Item

DEFAULT_EXTS = [".mp4", ".avi", ".mkv", ".webm", ".mov"]


def index_videos(root: Path, exts: list[str]) -> dict[str, Path]:
    """Map file stem → path for every video under ``root`` (recursive; first path in sorted order wins)."""
    idx: dict[str, Path] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            idx.setdefault(p.stem, p)
    return idx


def probe_decodable(path: Path) -> bool:
    """True if PyAV can decode the first frame and a frame near the end."""
    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            next(c.decode(stream))
            if c.duration:
                c.seek(int(c.duration * 0.9))
                next(c.decode(stream))
        return True
    except Exception:  # noqa: BLE001 — any failure means "undecodable"
        return False


def _candidates(item: Item) -> list[str]:
    v = item.video_id
    return [v, v[2:]] if v.startswith("v_") else [v]


def check_availability(items: list[Item], video_root: Path, probe: Callable[[Path], bool] = probe_decodable,
                       report_path: Path = Path("docs/data_availability.md"),
                       exts: list[str] = DEFAULT_EXTS) -> list[Item]:
    """Return items with ``video_path``/``available`` set; unavailable items are kept and reported."""
    idx = index_videos(video_root, exts)
    probed: dict[Path, bool] = {}
    out, problems = [], {}
    for it in items:
        path = next((idx[c] for c in _candidates(it) if c in idx), None)
        if path is None:
            problems[it.video_id] = "missing"
            out.append(it.model_copy(update={"video_path": "", "available": False}))
            continue
        if path not in probed:
            probed[path] = probe(path)
        ok = probed[path]
        if not ok:
            problems[it.video_id] = "undecodable"
        out.append(it.model_copy(update={"video_path": str(path), "available": ok}))
    _write_report(out, problems, Path(report_path))
    return out


def _write_report(items: list[Item], problems: dict[str, str], path: Path) -> None:
    by_bench = Counter((i.benchmark, i.available) for i in items)
    benches = sorted({i.benchmark for i in items})
    lines = ["# Data availability", "", "| benchmark | items available | items unavailable |", "|---|---|---|"]
    lines += [f"| {b} | {by_bench[(b, True)]} | {by_bench[(b, False)]} |" for b in benches]
    lines += ["", "| video_id | reason |", "|---|---|"]
    lines += [f"| {v} | {r} |" for v, r in sorted(problems.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
