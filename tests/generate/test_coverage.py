import itertools

from tests.generate.conftest import FakeFrameCache, FakeRenderer
from tests.helpers.factories import make_item
from vjaudit.generate.adapters import get_adapter
from vjaudit.generate.coverage import coverage, mc_accuracy, rerun_check, run_subset
from vjaudit.generate.engine import FakeEngine
from vjaudit.generate.runner import run_generation
from vjaudit.generate.systems import expand_mc_systems, expand_systems
from vjaudit.store import ArtifactStore


def _items():
    out = []
    for i in range(6):
        b = ["msvd", "nextoe"][i % 2]
        out.append(make_item(i, item_id=f"{b}:test:{i}", benchmark=b, in_main=True, in_annotated=i < 4,
                             mc_options=["a", "b", "c", "d", "e"] if b == "nextoe" else None,
                             mc_answer_idx=1 if b == "nextoe" else None))
    return out


def test_coverage_reports_missing_pairs(tmp_path, gens):
    store = ArtifactStore(tmp_path)
    items = _items()
    gen = next(g for g in gens if g.id == "qwen3vl8b")
    spec = next(s for s in expand_systems(gens, "msvd") if s.system_id == "qwen3vl8b")
    run_generation(store, items[:2], spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen,
                   get_adapter(gen.family, FakeRenderer()))
    cov = coverage(store, items, ["qwen3vl8b", "qwen3vl2b"])
    assert cov == {"qwen3vl8b": 4, "qwen3vl2b": 6}


def test_run_subset_annotated_then_main_skips_done(tmp_path, gens):
    store = ArtifactStore(tmp_path)
    items = _items()
    kw = dict(engine_factory=lambda g: FakeEngine(lambda r: "x"), renderer_factory=lambda g: FakeRenderer(),
              frame_cache=FakeFrameCache())
    n_ann = run_subset(store, items, gens, subset="annotated", **kw)
    assert n_ann == 4 * 11
    n_main = run_subset(store, items, gens, subset="main", **kw)
    assert n_main == 2 * 11
    assert all(v == 0 for v in coverage(store, items, [s.system_id for s in expand_systems(gens, "msvd")]).values())


def test_run_subset_sharding_partitions_generators(tmp_path, gens):
    items = _items()
    seen = []
    for k in range(3):
        store = ArtifactStore(tmp_path / str(k))
        run_subset(store, items, gens, subset="annotated", engine_factory=lambda g: FakeEngine(lambda r: "x"),
                   renderer_factory=lambda g: FakeRenderer(), frame_cache=FakeFrameCache(), shard=k, num_shards=3)
        seen.append({r["system_id"] for r in store.iter("generation")})
    assert not (seen[0] & seen[1]) and not (seen[1] & seen[2]) and len(set().union(*seen)) == 11


def test_mc_run_and_accuracy(tmp_path, gens):
    store = ArtifactStore(tmp_path)
    items = _items()
    answers = itertools.cycle(["B", "(C)", "none"])
    run_subset(store, items, gens, subset="main", mc=True, engine_factory=lambda g: FakeEngine(lambda r: next(answers)),
               renderer_factory=lambda g: FakeRenderer(), frame_cache=FakeFrameCache())
    acc = mc_accuracy(store, items, [s.system_id for s in expand_mc_systems(gens)])
    assert set(acc) == {s.system_id for s in expand_mc_systems(gens)}
    total_n = sum(v["n"] for v in acc.values())
    assert total_n == 3 * 5  # 3 nextoe items × 5 mc systems
    assert all(0 <= v["acc"] <= 1 and v["unparseable"] >= 0 for v in acc.values())


def test_rerun_check_detects_nondeterminism(tmp_path, gens):
    items = _items()
    gen = next(g for g in gens if g.id == "qwen3vl8b")
    spec = next(s for s in expand_systems(gens, "msvd") if s.system_id == "qwen3vl8b")
    adapter = get_adapter(gen.family, FakeRenderer())
    store = ArtifactStore(tmp_path / "main")
    run_generation(store, items, spec, FakeEngine(lambda r: "same"), FakeFrameCache(), gen, adapter)
    ok = rerun_check(store, items, spec, FakeEngine(lambda r: "same"), FakeFrameCache(), gen, adapter, n=4,
                     scratch=tmp_path / "s1")
    assert ok == (4, 4, [])
    counter = itertools.count()
    bad = rerun_check(store, items, spec, FakeEngine(lambda r: f"v{next(counter)}"), FakeFrameCache(), gen, adapter,
                      n=4, scratch=tmp_path / "s2")
    assert bad[0] == 0 and bad[1] == 4 and len(bad[2]) == 4
