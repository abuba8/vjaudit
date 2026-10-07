import json

from tests.annotate.helpers import gen, item
from vjaudit.annotate.export import export_claude, export_human
from vjaudit.annotate.units import build_units

SYSTEMS = ["qwen25vl3b", "qwen3vl2b", "llavaov05b"]


def _units_items():
    items = [item(1, question="what is in the video?"), item(2, question="what color | is it?")]
    gens = [gen("a", "msvd:test:1", SYSTEMS[0], "A dog.", "dog"), gen("b", "msvd:test:1", SYSTEMS[1], "a cat", "cat"),
            gen("c", "msvd:test:2", SYSTEMS[2], "red\nmaybe | blue", "red maybe blue"),
            gen("d", "msvd:test:2", SYSTEMS[0], "", "")]
    return build_units(items, gens), items


def test_claude_export_contains_rules_and_no_system_ids(tmp_path):
    units, items = _units_items()
    paths = export_claude(units, items, tmp_path)
    md = paths[0].read_text()
    assert paths[0].name == "batch_01.md" and "# VideoQA answer labelling — batch 01 of 01" in md
    assert "`unit_id,label,error_tags,note`" in md and "### Q1. what is in the video?" in md
    assert "Reference answer(s): dog | puppy" in md
    dog = next(u for u in units if u.answer_norm == "dog")
    assert f"| {dog.unit_id} | A dog. |" in md
    assert "red maybe \\| blue" in md and "(empty answer)" in md  # table-safe answers
    for f in tmp_path.iterdir():
        text = f.read_text().lower()
        assert not any(s in text for s in SYSTEMS) and "qwen" not in text and "llava" not in text
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {"batch_01": [u.unit_id for u in units]}


def test_human_export_header(tmp_path):
    units, items = _units_items()
    paths = export_human(units, items, tmp_path)
    lines = paths[0].read_text().splitlines()
    assert lines[0] == "unit_id,benchmark,question,reference_answers,candidate_answer,label,error_tags,note"
    assert lines[1] == f"{units[0].unit_id},msvd,what is in the video?,dog | puppy,{units[0].display_answer},,,"
    assert (tmp_path / "manifest.json").exists()


def test_reexport_adds_only_new_units_as_new_batches(tmp_path):
    units, items = _units_items()
    export_claude(units[:2], items, tmp_path)
    again = export_claude(units, items, tmp_path)
    assert [p.name for p in again] == ["batch_02.md"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["batch_01"] == [u.unit_id for u in units[:2]]
    assert manifest["batch_02"] == [u.unit_id for u in units[2:]]
    assert export_claude(units, items, tmp_path) == []  # nothing new
