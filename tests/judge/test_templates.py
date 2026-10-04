import hashlib
from pathlib import Path

import pytest

from vjaudit.judge.templates import (
    VCG_CONTENT_SHA256, load_template, protocol_id, render_rub, render_vcg, vendor_rub, verify_vcg_source,
)

UP = Path("tests/fixtures/upstream")


def test_vcg_template_hash_is_frozen():
    sys_t = Path("configs/prompts/judge/t_vcg_system.txt").read_bytes()
    usr_t = Path("configs/prompts/judge/t_vcg_user.txt").read_bytes()
    assert hashlib.sha256(sys_t + b"\x00" + usr_t).hexdigest() == VCG_CONTENT_SHA256


def test_vcg_files_match_upstream_source():
    assert verify_vcg_source([UP / "evaluate_activitynet_qa.py"]) == []


def test_verify_reports_upstream_differences():
    diffs = verify_vcg_source([UP / "evaluate_msvd_qa_variant.py"])
    assert len(diffs) == 1 and "system" in diffs[0]


def test_render_vcg_inserts_braces_quotes_newlines_literally():
    pred = "{'pred': 'no'}\nit's \"x\""
    msgs = render_vcg("what is {answer}?", "a {pred} b", pred)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    u = msgs[1]["content"]
    assert "Question: what is {answer}?\n" in u
    assert "Correct Answer: a {pred} b\n" in u
    assert f"Predicted Answer: {pred}\n\n" in u
    assert u.endswith("{'pred': 'yes', 'score': 4.8}.")


def test_render_rub_single_user_message_with_final_instruction():
    msgs = render_rub("why?", "to eat", "eating")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"].rstrip().endswith("Answer with a single word: Yes or No.")
    assert "why?" in msgs[0]["content"] and "to eat" in msgs[0]["content"] and "eating" in msgs[0]["content"]


def test_protocol_id_format():
    t = load_template("t-vcg")
    assert protocol_id(t, "t0") == f"t-vcg@{t.content_hash[:8]}/t0"
    assert protocol_id(load_template("t-rub"), "t0").startswith("t-rub@")


def test_vendor_rub_converts_prior_jinja(tmp_path):
    out = tmp_path / "t_rub.txt"
    vendor_rub(UP / "prior_judge_base.j2", {"q": "question", "gt": "answer", "gen": "pred"}, out)
    text = out.read_text()
    assert "Question: {question}" in text and "Ground truth: {answer}" in text and "Generated answer: {pred}" in text
    assert "Example:" not in text
    assert text.rstrip().endswith("Answer with a single word: Yes or No.")
