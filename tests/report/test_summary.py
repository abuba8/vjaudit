import pytest

from tests.helpers.factories import make_generation, make_item, make_judgecall
from vjaudit.hashing import short_id
from vjaudit.judge.templates import load_template, protocol_id
from vjaudit.report.summary import build_tables, judge_configs, visual_gain, write_summary
from vjaudit.schemas import Label
from vjaudit.store import ArtifactStore, stamp

J = [True, True, False, True]
R = [True, False, False, True]
CFG = "qwen3-4b/t-rub/t0"
RUB_NOW = protocol_id(load_template("t-rub"), "t0")  # the current T-RUB text's protocol id


def _label(item_id, norm, ok, source="claude-manual"):
    uid = short_id(item_id, norm)
    return Label(**stamp(dict(label_id=short_id(uid, source), unit_id=uid, source=source, task="ref",
                              label="correct" if ok else "incorrect", error_tags=[], note="", batch="batch_01"), {}))


def _fill(tmp_path, systems=("qwen25vl3b",), verdicts=None, labels=True):
    store = ArtifactStore(tmp_path)
    items = [make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True) for i in range(4)]
    for it in items:
        store.put("item", it, it.item_id)
    for s in systems:
        for i, it in enumerate(items):
            norm = f"{s}-ans{i}"
            g = make_generation(i, gen_id=f"{s}-{i}", item_id=it.item_id, system_id=s, answer_norm=norm)
            store.put("generation", g, g.gen_id)
            v = (verdicts or {}).get(s, J)[i]
            jc = make_judgecall(i, judge_call_id=f"jc-{s}-{i}", gen_id=g.gen_id, judge_id="qwen3-4b",
                                protocol_id=RUB_NOW, parsed_verdict=v,
                                parse_status="ok" if v is not None else "off_vocab")
            store.put("judgecall", jc, jc.judge_call_id)
            if labels:
                lab = _label(it.item_id, norm, R[i])
                store.put("label", lab, lab.label_id)
    return store, items


def test_judge_configs_keys(tmp_path):
    store, _ = _fill(tmp_path)
    extra = make_judgecall(9, judge_call_id="x", sample_idx=1, protocol_id="t-vcg@abcdef12/default-n3")
    assert judge_configs(list(store.iter("judgecall")) + [extra.model_dump()]) == [CFG]


def test_summary_tables_and_gap(tmp_path):
    store, items = _fill(tmp_path)
    row = build_tables(store, items, "annotated")["msvd"]["qwen25vl3b"]
    assert row["n"] == 4
    assert row["configs"][CFG] == {"acc": 0.75, "unparseable": 0, "judged": 4}
    assert row["sources"]["claude-manual"]["acc"] == 0.5
    agree = row["agreement"][CFG]
    assert agree["source"] == "claude-manual" and agree["gap_pts"] == pytest.approx(25.0)
    assert agree["kappa"] == pytest.approx(0.5) and agree["fpr"] == 0.5 and agree["fnr"] == 0.0 and agree["n"] == 4


def test_unparseable_verdicts_are_counted_not_mapped(tmp_path):
    store, items = _fill(tmp_path, verdicts={"qwen25vl3b": [True, None, False, True]})
    row = build_tables(store, items, "annotated")["msvd"]["qwen25vl3b"]
    assert row["configs"][CFG] == {"acc": pytest.approx(2 / 3), "unparseable": 1, "judged": 4}
    assert row["agreement"][CFG]["n"] == 3


def test_visual_gain(tmp_path):
    store, items = _fill(tmp_path, systems=("S", "S-blind"),
                         verdicts={"S": [True, True, True, False], "S-blind": [True, False, False, False]})
    gain = visual_gain(build_tables(store, items, "annotated"))
    assert gain["msvd"]["S"]["configs"][CFG] == pytest.approx(0.5)
    assert gain["msvd"]["S"]["sources"]["claude-manual"] == pytest.approx(0.0)
    assert "S-blind" not in gain["msvd"]


def test_summary_without_labels_still_writes(tmp_path):
    store, items = _fill(tmp_path, labels=False)
    out = tmp_path / "results.md"
    write_summary(store, items, "mac", out)
    text = out.read_text()
    assert "Label sources other than `human:*` are reference judges, not human ground truth." in text
    assert "## Run 1" in text and "## Run 2" in text and "| qwen25vl3b | 4 | 0.750 (0) |" in text
    assert "no labels imported yet" in text


def test_summary_with_labels_shows_agreement(tmp_path):
    store, items = _fill(tmp_path)
    out = tmp_path / "results.md"
    write_summary(store, items, "mac", out)
    text = out.read_text()
    assert "| qwen25vl3b | 4 | 0.750 (0) | 0.500 |" in text
    assert "| qwen25vl3b | qwen3-4b/t-rub/t0 | 4 | +25.0 | 0.500 | 0.500 | 0.000 |" in text


def test_verdicts_from_an_older_template_version_are_left_out_and_counted(tmp_path):
    store, items = _fill(tmp_path)
    old = make_judgecall(99, judge_call_id="old-0", gen_id="qwen25vl3b-0", judge_id="qwen3-4b",
                         protocol_id="t-rub@00000000/t0", parsed_verdict=False)
    store.put("judgecall", old, old.judge_call_id)
    tables = build_tables(store, items, "annotated")
    assert tables["msvd"]["qwen25vl3b"]["configs"][CFG]["acc"] == 0.75  # old verdict not mixed in
    out = tmp_path / "r.md"
    write_summary(store, items, "mac", out)
    assert "1 judge call(s) from an older version of a judge prompt are not counted" in out.read_text()


def test_partial_coverage_is_shown_in_the_cell(tmp_path):
    store, items = _fill(tmp_path)
    extra = make_item(9, item_id="msvd:test:9", in_main=True, in_annotated=False)
    store.put("item", extra, extra.item_id)
    g = make_generation(9, gen_id="qwen25vl3b-9", item_id=extra.item_id, system_id="qwen25vl3b", answer_norm="x")
    store.put("generation", g, g.gen_id)
    out = tmp_path / "r.md"
    write_summary(store, items + [extra], "mac", out)
    run2 = out.read_text().split("## Run 2")[1]
    assert "| qwen25vl3b | 5 | 0.750 (0) [4 of 5 judged] |" in run2
