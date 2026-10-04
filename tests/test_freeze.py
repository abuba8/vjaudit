import json

from tests.helpers.factories import make_generation, make_item, make_judgecall
from vjaudit.freeze import freeze
from vjaudit.store import ArtifactStore


def _fill(root):
    s = ArtifactStore(root)
    for i in range(3):
        s.put("item", make_item(i), f"msvd:test:{i}")
        g = make_generation(i)
        s.put("generation", g, g.gen_id)
        j = make_judgecall(i)
        s.put("judgecall", j, j.judge_call_id)
    return s


def test_freeze_is_deterministic(tmp_path):
    store = _fill(tmp_path / "a")
    cache = tmp_path / "cache" / "ab"
    cache.mkdir(parents=True)
    (cache / "ab12.json").write_text('{"request": {}, "response": {}}')
    m1 = freeze(store, tmp_path / "cache", tmp_path / "out1")
    m2 = freeze(ArtifactStore(tmp_path / "a"), tmp_path / "cache", tmp_path / "out2")
    strip = lambda m: [(e.kind, e.n_records, e.sha256) for e in m]
    assert strip(m1) == strip(m2)
    kinds = {e.kind: e for e in m1}
    assert {"item", "generation", "judgecall", "api_cache"} == set(kinds)
    assert kinds["judgecall"].n_records == 3 and kinds["api_cache"].n_records == 1


def test_freeze_writes_manifest_jsonl(tmp_path):
    store = _fill(tmp_path / "a")
    (tmp_path / "cache").mkdir()
    freeze(store, tmp_path / "cache", tmp_path / "out", manifest=tmp_path / "manifest.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"item", "generation", "judgecall", "api_cache"}
    assert all(len(r["sha256"]) == 64 for r in rows)
