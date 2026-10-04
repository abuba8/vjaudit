import json
import logging
from pathlib import Path

import av
import numpy as np
import pytest

from vjaudit.data.availability import check_availability, index_videos, probe_decodable
from vjaudit.data.loaders import load_benchmark, qtype_msvd

RAW = Path("tests/fixtures/raw")


def test_msvd_item_ids_and_qtypes():
    items = load_benchmark("msvd", RAW / "msvd")
    by_id = {i.item_id: i for i in items}
    assert by_id["msvd:test:3"].qtype == "where"
    assert by_id["msvd:test:3"].video_id == "q-r_0_8"
    assert by_id["msvd:test:0"].answers_ref == ["dog"]
    assert by_id["msvd:test:5"].qtype == "other"
    assert all(i.benchmark == "msvd" and i.split == "test" for i in items)


@pytest.mark.parametrize("q,t", [("What is it?", "what"), ("who runs", "who"), ("How many?", "how"),
                                 ("when?", "when"), ("Where is", "where"), ("is it red", "other")])
def test_qtype_msvd(q, t):
    assert qtype_msvd(q) == t


def test_anetqa_join_drops_unmatched_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        items = load_benchmark("anetqa", RAW / "anetqa")
    assert {i.item_id for i in items} == {"anetqa:test:v_aaa_0", "anetqa:test:v_aaa_1", "anetqa:test:v_bbb_0"}
    assert "v_ccc_0" in caplog.text
    by_id = {i.item_id: i for i in items}
    assert by_id["anetqa:test:v_aaa_0"].qtype == "color"
    assert by_id["anetqa:test:v_bbb_0"].qtype == "motion"
    assert by_id["anetqa:test:v_aaa_0"].video_id == "v_aaa"


def test_nextoe_mc_join_attaches_options():
    items = {i.item_id: i for i in load_benchmark("nextoe", RAW / "nextqa")}
    a = items["nextoe:test:1001_0"]
    assert a.answers_ref == ["lost balance", "slipped"]
    assert len(a.mc_options) == 5 and a.mc_answer_idx == 2 and a.mc_options[2] == "lost balance"
    b = items["nextoe:test:1001_1"]
    assert b.mc_options is None and b.mc_answer_idx is None and b.qtype == "DB"
    assert items["nextoe:test:1002_3"].video_id == "1002"


def test_duplicate_native_qid_raises(tmp_path):
    d = tmp_path / "msvd"
    d.mkdir()
    (d / "youtube_mapping.txt").write_text("c1 vid1\n")
    (d / "test_qa.json").write_text(json.dumps([
        {"answer": "a", "id": 7, "question": "what?", "video_id": 1},
        {"answer": "b", "id": 7, "question": "who?", "video_id": 1}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark("msvd", d)


def _tiny_video(path: Path, n: int = 10):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 20, np.uint8), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_index_videos_recursive_by_stem(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "v_aaa.mp4").write_bytes(b"x")
    (tmp_path / "zzz.txt").write_text("x")
    idx = index_videos(tmp_path, [".mp4", ".avi"])
    assert idx == {"v_aaa": tmp_path / "a" / "b" / "v_aaa.mp4"}


def test_probe_decodable_real_and_corrupt(tmp_path):
    good = tmp_path / "good.mp4"
    _tiny_video(good)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00not a video" * 50)
    assert probe_decodable(good) is True
    assert probe_decodable(bad) is False


def test_unavailable_video_is_flagged_not_dropped(tmp_path):
    items = load_benchmark("anetqa", RAW / "anetqa")
    vids = tmp_path / "videos"
    vids.mkdir()
    (vids / "v_aaa.mp4").write_bytes(b"x")  # present + "decodable" per fake probe
    (vids / "bbb.mp4").write_bytes(b"x")  # present without v_ prefix, but probe says undecodable
    report = tmp_path / "data_availability.md"
    out = check_availability(items, vids, probe=lambda p: p.name == "v_aaa.mp4", report_path=report,
                             exts=[".mp4"])
    assert len(out) == len(items)
    by_id = {i.item_id: i for i in out}
    assert by_id["anetqa:test:v_aaa_0"].available and by_id["anetqa:test:v_aaa_0"].video_path.endswith("v_aaa.mp4")
    assert not by_id["anetqa:test:v_bbb_0"].available
    text = report.read_text()
    assert "v_bbb" in text and "undecodable" in text


def test_missing_videos_reported_missing(tmp_path):
    items = load_benchmark("msvd", RAW / "msvd")
    (tmp_path / "videos").mkdir()
    report = tmp_path / "r.md"
    out = check_availability(items, tmp_path / "videos", probe=lambda p: True, report_path=report)
    assert not any(i.available for i in out)
    assert "missing" in report.read_text() and "-abc_1_5" in report.read_text()
