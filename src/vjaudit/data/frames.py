"""Uniform, segment-centred frame sampling with an on-disk cache.

Generators get ``n`` frames at ``sample_indices(...)``. VLM judges (later plans) use the
documented subset ``indices[::2]`` of the generator's frames (R-EXP-06).
"""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image


class FrameDecodeError(RuntimeError):
    pass


def sample_indices(n_total: int, n: int, start: int = 0, end: int | None = None) -> list[int]:
    """[start + int((i + 0.5) * L / n) for i in range(n)], L = (end or n_total) - start."""
    length = (end if end is not None else n_total) - start
    return [start + int((i + 0.5) * length / n) for i in range(n)]


def _frame_count_and_fps(path: Path) -> tuple[int, float]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 0) or 1.0
        n = s.frames
        if not n:
            n = sum(1 for _ in c.decode(s))
    return n, fps


def _decode_frames(path: Path, wanted: list[int], max_side: int) -> np.ndarray:
    want = set(wanted)
    got: dict[int, np.ndarray] = {}
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                img = frame.to_image()
                img.thumbnail((max_side, max_side), Image.BICUBIC)
                got[i] = np.asarray(img.convert("RGB"), dtype=np.uint8)
            if i >= max(wanted):
                break
    missing = [i for i in wanted if i not in got]
    if missing:
        raise FrameDecodeError(f"frames {missing[:5]} not decodable in {path}")
    return np.stack([got[i] for i in wanted])


class FrameCache:
    def __init__(self, root: Path | str, max_side: int = 768):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_side = max_side

    def _key(self, video_id: str, n: int, t_start: float | None, t_end: float | None) -> Path:
        safe = video_id.replace("/", "_")
        return self.root / f"{safe}_{n}_centered_{t_start}_{t_end}_{self.max_side}.npz"

    def get(self, video_path: Path | str, video_id: str, n: int, t_start: float | None,
            t_end: float | None) -> tuple[np.ndarray, list[int]]:
        """Return (uint8 array (n, H, W, 3), frame indices); decode once, then serve from cache."""
        key = self._key(video_id, n, t_start, t_end)
        if key.exists():
            z = np.load(key)
            return z["frames"], z["indices"].tolist()
        try:
            total, fps = _frame_count_and_fps(Path(video_path))
            start = int(t_start * fps) if t_start is not None else 0
            end = min(int(t_end * fps), total) if t_end is not None else total
            idx = sample_indices(total, n, start, end)
            arr = _decode_frames(Path(video_path), idx, self.max_side)
        except FrameDecodeError as e:
            raise FrameDecodeError(f"{video_id}: {e}") from e
        except Exception as e:  # noqa: BLE001 — PyAV raises many types; always name the video
            raise FrameDecodeError(f"{video_id}: cannot decode {video_path}: {e}") from e
        tmp = key.with_suffix(".tmp.npz")
        np.savez(tmp, frames=arr, indices=np.asarray(idx))
        tmp.replace(key)
        return arr, idx
