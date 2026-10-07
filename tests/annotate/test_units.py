from tests.annotate.helpers import gen, item
from vjaudit.annotate.units import build_units, plan_batches
from vjaudit.hashing import short_id


def test_units_dedup_by_normalized_answer():
    items = [item(1)]
    gens = [gen("b2", "msvd:test:1", "qwen3vl2b", "A dog.", "dog"),
            gen("a1", "msvd:test:1", "qwen25vl3b", "dog", "dog"),
            gen("c3", "msvd:test:1", "llavaov05b", "cat", "cat")]
    units = build_units(items, gens)
    assert len(units) == 2
    dog = next(u for u in units if u.answer_norm == "dog")
    assert dog.unit_id == short_id("msvd:test:1", "dog") and dog.benchmark == "msvd"
    assert dog.display_answer == "dog"  # output_text of the smallest gen_id ("a1")
    assert sorted(dog.system_ids) == ["qwen25vl3b", "qwen3vl2b"] and sorted(dog.gen_ids) == ["a1", "b2"]


def test_units_skip_items_outside_the_set_and_mc_answers():
    gens = [gen("a", "msvd:test:1", "s", "dog", "dog"), gen("b", "msvd:test:9", "s", "cat", "cat"),
            {**gen("c", "msvd:test:1", "s-mc", "B", "b"), "style": "mc"}]
    units = build_units([item(1)], gens)
    assert [u.answer_norm for u in units] == ["dog"]


def test_batches_never_split_a_question():
    from vjaudit.annotate.units import Unit
    def units_for(q, n):
        return [Unit(unit_id=f"{q}-{k}", item_id=q, benchmark="msvd", answer_norm=str(k), display_answer=str(k),
                     gen_ids=[], system_ids=[], config_hash="x", git_sha="x", created_at="x") for k in range(n)]
    units = units_for("q1", 60) + units_for("q2", 50) + units_for("q3", 10)
    batches = plan_batches(units, batch_size=100)
    assert [[u.item_id for u in b][0] for b in batches] == ["q1", "q2"]
    assert [len(b) for b in batches] == [60, 60]


def test_a_question_larger_than_the_batch_gets_its_own_batch():
    from vjaudit.annotate.units import Unit
    units = [Unit(unit_id=f"u{k}", item_id="q1" if k < 5 else "q2", benchmark="msvd", answer_norm=str(k),
                  display_answer=str(k), gen_ids=[], system_ids=[], config_hash="x", git_sha="x", created_at="x")
             for k in range(7)]
    assert [len(b) for b in plan_batches(units, batch_size=3)] == [5, 2]
