import pytest

from vjaudit.annotate.importer import LabelImportError, import_labels, parse_label_csv, validate_source
from vjaudit.store import ArtifactStore

MANIFEST = {"batch_01": ["u1", "u2", "u3", "u4"], "batch_02": ["u5"]}
GOOD = "unit_id,label,error_tags,note\nu1,correct,,\nu2,incorrect,E2;E4,too general\nu3,Correct,,\"ok, fine\"\n"


def test_parse_fenced_crlf_quoted_reply():
    text = "Sure!\r\n```csv\r\nunit_id,label,error_tags,note\r\nu1,correct,,\"ok, fine\"\r\n```\r\nDone"
    assert parse_label_csv(text) == [{"row": 1, "unit_id": "u1", "label": "correct", "error_tags": [],
                                      "note": "ok, fine"}]


def test_parse_plain_csv_and_tags():
    rows = parse_label_csv("﻿" + GOOD)
    assert [r["unit_id"] for r in rows] == ["u1", "u2", "u3"]
    assert rows[1]["error_tags"] == ["E2", "E4"] and rows[2]["label"] == "correct"


def test_parse_without_header_is_an_error():
    with pytest.raises(LabelImportError, match="unit_id"):
        parse_label_csv("u1,correct,,\n")


def test_import_rejects_invented_ids_and_bad_labels_atomically(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = parse_label_csv("unit_id,label,error_tags,note\nzzz,correct,,\nu1,maybe,,\nu2,incorrect,E99,\nu3,correct,,\n")
    with pytest.raises(LabelImportError) as ei:
        import_labels(store, rows, "claude-manual", MANIFEST)
    assert len(ei.value.errors) == 3
    assert "row 1" in ei.value.errors[0] and "zzz" in ei.value.errors[0]
    assert "maybe" in ei.value.errors[1] and "E99" in ei.value.errors[2]
    assert list(store.iter("label")) == []


def test_import_rejects_conflicting_duplicates(tmp_path):
    rows = parse_label_csv("unit_id,label,error_tags,note\nu1,correct,,\nu1,incorrect,,\n")
    with pytest.raises(LabelImportError, match="u1"):
        import_labels(ArtifactStore(tmp_path), rows, "claude-manual", MANIFEST)


def test_import_reports_missing_units_per_batch(tmp_path):
    store = ArtifactStore(tmp_path)
    res = import_labels(store, parse_label_csv(GOOD), "claude-manual", MANIFEST)
    assert res.stored == 3 and res.skipped_existing == 0
    assert res.missing_by_batch == {"batch_01": 1, "batch_02": 1}
    labels = {r["unit_id"]: r for r in store.iter("label")}
    assert labels["u2"]["error_tags"] == ["E2", "E4"] and labels["u2"]["batch"] == "batch_01"
    assert labels["u1"]["source"] == "claude-manual" and labels["u1"]["task"] == "ref"


def test_reimport_skips_existing(tmp_path):
    store = ArtifactStore(tmp_path)
    assert import_labels(store, parse_label_csv(GOOD), "claude-manual", MANIFEST).stored == 3
    again = import_labels(store, parse_label_csv(GOOD), "claude-manual", MANIFEST)
    assert again.stored == 0 and again.skipped_existing == 3
    v2 = import_labels(store, parse_label_csv(GOOD), "claude-manual-v2", MANIFEST)
    assert v2.stored == 3  # a corrected pass goes in under a new source


@pytest.mark.parametrize("src,ok", [("claude-manual", True), ("claude-manual-v2", True), ("human:ann1", True),
                                    ("human", False), ("claude", False), ("gpt", False), ("human:", False)])
def test_validate_source(src, ok):
    if ok:
        validate_source(src)
    else:
        with pytest.raises(ValueError):
            validate_source(src)


def test_parse_unfenced_reply_ignores_trailing_prose():
    text = "Here are the labels.\nunit_id,label,error_tags,note\nu1,correct,,\nu2,incorrect,E2,\n\nLet me know if you need more.\n"
    assert [r["unit_id"] for r in parse_label_csv(text)] == ["u1", "u2"]


def test_parse_header_with_spaces_after_commas():
    rows = parse_label_csv("unit_id, label, error_tags, note\nu1, correct, E1;E2, fine\n")
    assert rows == [{"row": 1, "unit_id": "u1", "label": "correct", "error_tags": ["E1", "E2"], "note": "fine"}]


def test_row_with_too_many_fields_is_an_error(tmp_path):
    rows = parse_label_csv("unit_id,label,error_tags,note\nu2,incorrect,E2, E4,unquoted, note\n")
    with pytest.raises(LabelImportError, match="row 1.*more than 4 fields"):
        import_labels(ArtifactStore(tmp_path), rows, "claude-manual", MANIFEST)
