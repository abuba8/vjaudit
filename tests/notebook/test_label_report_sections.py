"""Sections 10–11: labelling and report, pinned to the package's annotate and report modules."""
from pathlib import Path

import pandas as pd
import pytest

from tests.annotate.helpers import gen, item
from tests.notebook.conftest import read_notebook
from vjaudit.annotate.export import export_claude, export_human
from vjaudit.annotate.importer import LabelImportError, import_labels, parse_label_csv
from vjaudit.annotate.units import build_units, plan_batches
from vjaudit.report.metrics import cohen_kappa, error_rates
from vjaudit.store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "configs" / "prompts"


def _data():
    items = [item(1, question="what is in the video?"), item(2, question="what color | is it?"), item(3)]
    gens = [gen("a", "msvd:test:1", "s1", "A dog.", "dog"), gen("b", "msvd:test:1", "s2", "a cat", "cat"),
            gen("c", "msvd:test:2", "s3", "red\nmaybe | blue", "red maybe blue"), gen("d", "msvd:test:2", "s1", "", ""),
            gen("e", "msvd:test:1", "s3", "dog", "dog"), gen("f", "msvd:test:9", "s1", "x", "x")]
    return items, gens


def _nb_frames(items, gens):
    return pd.DataFrame([i.model_dump() for i in items]), pd.DataFrame(gens)


def test_units_and_batches_match_package(nb_lib):
    items, gens = _data()
    theirs = build_units(items, gens)
    ours = nb_lib["build_units"](*_nb_frames(items, gens))
    assert [(u.unit_id, u.item_id, u.answer_norm, u.display_answer, sorted(u.gen_ids), sorted(u.system_ids))
            for u in theirs] == [(r["unit_id"], r["item_id"], r["answer_norm"], r["display_answer"],
                                  sorted(r["gen_ids"]), sorted(r["system_ids"])) for r in ours.to_dict("records")]
    for size in (1, 2, 100):
        assert [[u.unit_id for u in b] for b in plan_batches(theirs, size)] == \
            [list(b["unit_id"]) for b in nb_lib["plan_batches"](ours, size)]


def test_claude_and_human_exports_match_package(nb_lib, tmp_path):
    items, gens = _data()
    theirs = build_units(items, gens)
    idf, gdf = _nb_frames(items, gens)
    ours = nb_lib["build_units"](idf, gdf)
    a = export_claude(theirs, items, tmp_path / "pkg", batch_size=2)
    b = nb_lib["export_claude"](ours, idf, tmp_path / "nb", batch_size=2)
    assert [p.name for p in a] == [p.name for p in b]
    for pa, pb in zip(a, b):
        assert pa.read_text() == pb.read_text()
    assert (tmp_path / "pkg" / "manifest.json").read_text() == (tmp_path / "nb" / "manifest.json").read_text()
    h1 = export_human(theirs, items, tmp_path / "pkg-h")
    h2 = nb_lib["export_human"](ours, idf, tmp_path / "nb-h")
    assert h1[0].read_text() == h2[0].read_text()
    assert nb_lib["export_claude"](ours, idf, tmp_path / "nb", batch_size=2) == []  # nothing new


REPLIES = [
    "Sure!\r\n```csv\r\nunit_id,label,error_tags,note\r\nu1,correct,,\"ok, fine\"\r\n```\r\nDone",
    "Here are the labels.\nunit_id,label,error_tags,note\nu1,correct,,\nu2,incorrect,E2,\n\nLet me know.\n",
    "unit_id, label, error_tags, note\nu1, correct, E1;E2, fine\n",
    "﻿unit_id,label,error_tags,note\nu1,Correct,e2; E4 ,x\n\nu2,incorrect,,\n",
    "unit_id,label,error_tags,note\nu2,incorrect,E2, E4,unquoted, note\n",
]


@pytest.mark.parametrize("text", REPLIES)
def test_parse_label_csv_matches_package(nb_lib, text):
    assert nb_lib["parse_label_csv"](text) == parse_label_csv(text)


def test_parse_without_header_raises(nb_lib):
    with pytest.raises(nb_lib["LabelImportError"], match="unit_id"):
        nb_lib["parse_label_csv"]("u1,correct,,\n")


MANIFEST = {"batch_01": ["u1", "u2", "u3", "u4"], "batch_02": ["u5"]}
GOOD = "unit_id,label,error_tags,note\nu1,correct,,\nu2,incorrect,E2;E4,too general\nu3,Correct,,\"ok, fine\"\n"


def test_import_validation_matches_package(nb_lib, tmp_path):
    bad = "unit_id,label,error_tags,note\nzzz,correct,,\nu1,maybe,,\nu2,incorrect,E99,\nu3,correct,,\nu3,incorrect,,\n"
    with pytest.raises(LabelImportError) as theirs:
        import_labels(ArtifactStore(tmp_path / "p"), parse_label_csv(bad), "claude-manual", MANIFEST)
    with pytest.raises(nb_lib["LabelImportError"]) as ours:
        nb_lib["import_labels"](nb_lib["Store"](tmp_path / "n"), nb_lib["parse_label_csv"](bad), "claude-manual", MANIFEST)
    assert ours.value.errors == theirs.value.errors
    assert nb_lib["Store"](tmp_path / "n").load("labels").empty


def test_import_stores_skips_and_reports_missing(nb_lib, tmp_path):
    st = nb_lib["Store"](tmp_path)
    r = nb_lib["import_labels"](st, nb_lib["parse_label_csv"](GOOD), "claude-manual", MANIFEST)
    assert r == {"stored": 3, "skipped_existing": 0, "missing_by_batch": {"batch_01": 1, "batch_02": 1}}
    again = nb_lib["import_labels"](st, nb_lib["parse_label_csv"](GOOD), "claude-manual", MANIFEST)
    assert again["stored"] == 0 and again["skipped_existing"] == 3
    labels = st.load("labels").set_index("unit_id")
    assert labels.loc["u2", "error_tags"] == ["E2", "E4"] and labels.loc["u2", "batch"] == "batch_01"
    for bad_source in ("human", "claude", "gpt", "human:"):
        with pytest.raises(ValueError):
            nb_lib["validate_source"](bad_source)


@pytest.mark.parametrize("j,r", [([True, True, False, True], [True, False, False, True]),
                                 ([True] * 3, [True] * 3), ([True, False], [False, True]), ([], [])])
def test_metrics_match_package(nb_lib, j, r):
    assert nb_lib["cohen_kappa"](j, r) == cohen_kappa(j, r)
    if j:
        assert nb_lib["error_rates"](j, r) == error_rates(j, r)


def _report_store(nb_lib, tmp_path, labels=True):
    st = nb_lib["Store"](tmp_path)
    J, R = [True, True, False, True], [True, False, False, True]
    rub = nb_lib["protocol_id"]("t-rub", "t0", PROMPTS)
    items = [dict(item(i).model_dump(), item_id=f"msvd:test:{i}", available=True) for i in range(4)]
    st.append("items", items, key="item_id")
    for s in ("S", "S-blind"):
        for i in range(4):
            norm = f"{s}-ans{i}"
            st.append("generations", [dict(gen_id=f"{s}-{i}", item_id=f"msvd:test:{i}", benchmark="msvd", system_id=s,
                                           style="short", output_text=norm, answer_norm=norm)], key="gen_id")
            v = J[i] if s == "S" else False
            st.append("judgecalls", [dict(judge_call_id=f"jc-{s}-{i}", gen_id=f"{s}-{i}", judge_id="qwen3-4b",
                                          protocol_id=rub, sample_idx=0, parsed_verdict=v, parse_status="ok")],
                      key="judge_call_id")
            if labels:
                uid = nb_lib["short_id"](f"msvd:test:{i}", norm)
                st.append("labels", [dict(label_id=nb_lib["short_id"](uid, "claude-manual"), unit_id=uid,
                                          source="claude-manual", task="ref",
                                          label="correct" if (R[i] and s == "S") else "incorrect",
                                          error_tags=[], note="", batch="batch_01")], key="label_id")
    return st, pd.DataFrame(items)


def test_build_tables_gap_kappa_and_visual_gain(nb_lib, tmp_path):
    st, items = _report_store(nb_lib, tmp_path)
    tables = nb_lib["build_tables"](st, items, "annotated", PROMPTS)
    row = tables["msvd"]["S"]
    assert row["n"] == 4 and row["configs"]["qwen3-4b/t-rub/t0"] == {"acc": 0.75, "unparseable": 0, "judged": 4}
    assert row["sources"]["claude-manual"]["acc"] == 0.5
    agree = row["agreement"]["qwen3-4b/t-rub/t0"]
    assert agree["gap_pts"] == pytest.approx(25.0) and agree["kappa"] == pytest.approx(0.5)
    gain = nb_lib["visual_gain"](tables)
    assert gain["msvd"]["S"]["configs"]["qwen3-4b/t-rub/t0"] == pytest.approx(0.75)


def test_results_md_written(nb_lib, tmp_path):
    st, items = _report_store(nb_lib, tmp_path)
    out = nb_lib["write_results"](st, items, "mac", PROMPTS, tmp_path / "results.md")
    text = out.read_text()
    assert "Label sources other than `human:*` are reference judges" in text
    assert "Agreement with `claude-manual`" in text and "| S | qwen3-4b/t-rub/t0 | 4 | +25.0 | 0.500 |" in text
    st2, items2 = _report_store(nb_lib, tmp_path / "nolabels", labels=False)
    assert "no labels imported yet" in nb_lib["write_results"](st2, items2, "mac", PROMPTS, tmp_path / "r2.md").read_text()


def test_notebook_has_all_sections():
    headers = [line for c in read_notebook().cells if c.cell_type == "markdown"
               for line in c.source.splitlines() if line.startswith("## ")]
    assert [h.split(".")[0][3:] for h in headers] == [str(k) for k in range(13)]
