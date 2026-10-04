from pathlib import Path

import av
import numpy as np
import pytest

from vjaudit.data import frames as frames_mod
from vjaudit.data.frames import FrameCache, FrameDecodeError, sample_indices


def _video(path: Path, n: int = 40, w: int = 32, h: int = 32, fps: int = 10):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"qscale": "1"}
        for i in range(n):
            img = np.full((h, w, 3), i * 6, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def test_sample_indices_formula():
    assert sample_indices(40, 4) == [5, 15, 25, 35]
    assert sample_indices(40, 4, start=20, end=40) == [22, 27, 32, 37]
    assert sample_indices(16, 16) == list(range(16))


def test_cache_returns_exact_frames(tmp_path):
    v = _video(tmp_path / "v.mp4")
    arr, idx = FrameCache(tmp_path / "cache").get(v, "v", 4, None, None)
    assert idx == [5, 15, 25, 35] and arr.shape == (4, 32, 32, 3) and arr.dtype == np.uint8
    for k, i in enumerate(idx):
        assert abs(int(arr[k, 16, 16, 0]) - 6 * i) <= 4


def test_cache_hit_is_identical_and_skips_decode(tmp_path, monkeypatch):
    v = _video(tmp_path / "v.mp4")
    cache = FrameCache(tmp_path / "cache")
    a1, i1 = cache.get(v, "v", 4, None, None)
    monkeypatch.setattr(frames_mod, "_decode_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    a2, i2 = cache.get(v, "v", 4, None, None)
    assert i1 == i2 and np.array_equal(a1, a2)


def test_segment_restricts_range(tmp_path):
    v = _video(tmp_path / "v.mp4")  # 10 fps, 40 frames = 4 s
    _, idx = FrameCache(tmp_path / "c").get(v, "v", 4, 2.0, 4.0)
    assert idx == [22, 27, 32, 37]


def test_corrupt_file_raises_with_video_id(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00garbage" * 100)
    with pytest.raises(FrameDecodeError, match="vid_bad"):
        FrameCache(tmp_path / "c").get(bad, "vid_bad", 4, None, None)


def test_resize_preserves_aspect_and_max_side(tmp_path):
    v = _video(tmp_path / "wide.mp4", n=8, w=64, h=32)
    arr, _ = FrameCache(tmp_path / "c", max_side=16).get(v, "wide", 2, None, None)
    assert arr.shape == (2, 8, 16, 3)
