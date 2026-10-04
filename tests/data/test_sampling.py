from collections import Counter

import pytest

from tests.helpers.factories import make_item
from vjaudit.data.sampling import allocate, assign_nested, split_dev_test, stratified_sample


def _pool():
    items, n = [], 0
    for bench, mix in {"msvd": {"what": 3000, "who": 1500, "how": 100, "when": 40, "where": 30},
                       "anetqa": {"motion": 900, "color": 700, "yes_no": 1400, "other": 600}}.items():
        for qt, k in mix.items():
            for _ in range(k):
                items.append(make_item(n, item_id=f"{bench}:test:{n}", benchmark=bench, qtype=qt))
                n += 1
    items.append(make_item(n, item_id=f"msvd:test:{n}", benchmark="msvd", qtype="what", available=False))
    return items


def test_allocate_sums_to_n_and_respects_min():
    a = allocate({"a": 900, "b": 90, "c": 10}, 100, min_per=5)
    assert sum(a.values()) == 100 and a["c"] >= 5 and a["b"] >= 5
    assert allocate({"a": 3, "b": 1}, 4) == {"a": 3, "b": 1}
    assert allocate({"a": 2, "b": 50}, 10, min_per=5)["a"] == 2  # capped at stratum size


def test_stratified_sample_size_and_determinism():
    pool = [i for i in _pool() if i.benchmark == "anetqa"]
    s1 = stratified_sample(pool, 100, seed=42)
    s2 = stratified_sample(pool, 100, seed=42)
    assert len(s1) == 100 and [i.item_id for i in s1] == [i.item_id for i in s2]
    assert [i.item_id for i in stratified_sample(pool, 100, seed=7)] != [i.item_id for i in s1]


@pytest.fixture(scope="module")
def nested():
    return assign_nested(_pool(), sizes=(2000, 300, 200), seed=42, msvd_min_per=15)


def test_nesting_holds(nested):
    for i in nested:
        if i.in_annotated:
            assert i.in_main
        if i.in_vid_subset:
            assert i.in_annotated and i.vid_rank is not None
    for bench in ("msvd", "anetqa"):
        c = Counter((i.in_main, i.in_annotated, i.in_vid_subset) for i in nested if i.benchmark == bench)
        assert sum(v for k, v in c.items() if k[0]) == 2000
        assert sum(v for k, v in c.items() if k[1]) == 300
        assert sum(v for k, v in c.items() if k[2]) == 200


def test_msvd_min_15_per_type(nested):
    c = Counter(i.qtype for i in nested if i.benchmark == "msvd" and i.in_annotated)
    assert all(c[q] >= 15 for q in ("what", "who", "how", "when", "where"))


def test_unavailable_items_never_sampled(nested):
    assert not any(i.in_main for i in nested if not i.available)


def test_vid_rank_prefix_covers_all_qtypes(nested):
    vid = sorted((i for i in nested if i.benchmark == "msvd" and i.in_vid_subset), key=lambda i: i.vid_rank)
    assert sorted(i.vid_rank for i in vid) == list(range(len(vid)))
    have = Counter(i.qtype for i in vid)
    prefix = {i.qtype for i in vid[:150]}
    assert {q for q, k in have.items() if k >= 3} <= prefix


def test_dev_test_is_40_60_within_1pct_per_benchmark(nested):
    split = split_dev_test(nested, frac_dev=0.4, seed=42)
    for bench in ("msvd", "anetqa"):
        ann = [i for i in split if i.benchmark == bench and i.in_annotated]
        dev = sum(i.human_split == "dev" for i in ann)
        assert abs(dev / len(ann) - 0.4) <= 0.01
        assert all(i.human_split in ("dev", "test") for i in ann)
    assert all(i.human_split is None for i in split if not i.in_annotated)


def test_same_seed_same_sample():
    a = assign_nested(_pool(), seed=42)
    b = assign_nested(_pool(), seed=42)
    key = lambda xs: [(i.item_id, i.in_main, i.in_annotated, i.in_vid_subset, i.vid_rank) for i in xs]
    assert key(a) == key(b)
