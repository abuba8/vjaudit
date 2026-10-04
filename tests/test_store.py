import json
import logging

import pytest

from tests.helpers.factories import make_generation
from vjaudit.store import ArtifactStore, stamp


def test_put_is_idempotent(tmp_path):
    s = ArtifactStore(tmp_path)
    g = make_generation()
    assert s.put("generation", g, g.gen_id) is True
    assert s.put("generation", g, g.gen_id) is False
    assert len(list(s.iter("generation"))) == 1
    assert s.has("generation", g.gen_id)


def test_keys_survive_reopen(tmp_path):
    s = ArtifactStore(tmp_path)
    g = make_generation()
    s.put("generation", g, g.gen_id)
    s2 = ArtifactStore(tmp_path)
    assert s2.has("generation", g.gen_id)
    assert s2.put("generation", g, g.gen_id) is False


def test_truncated_last_line_is_ignored_and_reported(tmp_path, caplog):
    s = ArtifactStore(tmp_path)
    for i in range(3):
        g = make_generation(i)
        s.put("generation", g, g.gen_id)
    part = next((tmp_path / "generation").glob("*.jsonl"))
    with open(part, "a") as f:
        f.write('{"key": "half", "record": {"gen_')
    with caplog.at_level(logging.WARNING):
        s2 = ArtifactStore(tmp_path)
        rows = list(s2.iter("generation"))
    assert len(rows) == 3
    assert part.name in caplog.text


def test_corrupt_middle_line_raises(tmp_path):
    d = tmp_path / "generation"
    d.mkdir()
    (d / "part-1-0.jsonl").write_text('not json\n{"key": "a", "record": {}}\n')
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path)


def _job(store, keys, crash_after=None):
    for n, k in enumerate(keys):
        if crash_after is not None and n == crash_after:
            raise RuntimeError("simulated crash")
        if store.has("generation", f"{k:016x}"):
            continue
        g = make_generation(k)
        store.put("generation", g, g.gen_id)


def test_interrupt_then_resume_equals_uninterrupted(tmp_path):
    keys = list(range(50))
    clean = ArtifactStore(tmp_path / "clean")
    _job(clean, keys)
    h_clean = clean.export_sorted("generation", tmp_path / "clean.jsonl")

    crashy = ArtifactStore(tmp_path / "crashy")
    with pytest.raises(RuntimeError):
        _job(crashy, keys, crash_after=20)
    _job(ArtifactStore(tmp_path / "crashy"), keys)
    h_resumed = ArtifactStore(tmp_path / "crashy").export_sorted("generation", tmp_path / "resumed.jsonl")
    assert h_clean == h_resumed


def test_export_sorted_drops_created_at(tmp_path):
    s = ArtifactStore(tmp_path)
    g = make_generation()
    s.put("generation", g, g.gen_id)
    s.export_sorted("generation", tmp_path / "out.jsonl")
    rec = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    assert "created_at" not in rec["record"] and rec["key"] == g.gen_id


def test_stamp_adds_provenance_fields():
    out = stamp({"x": 1}, {"n_frames": 16})
    assert set(out) == {"x", "config_hash", "git_sha", "created_at"}
    assert len(out["config_hash"]) == 64 and out["created_at"].endswith("+00:00")
