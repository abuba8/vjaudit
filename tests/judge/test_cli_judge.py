import shutil

import yaml
from typer.testing import CliRunner

from vjaudit.cli import app


def test_verify_vcg_marks_source_verified(tmp_path):
    src = tmp_path / "source.yaml"
    shutil.copy("configs/prompts/judge/source.yaml", src)
    res = CliRunner().invoke(app, ["judge", "verify-vcg", "--upstream", "tests/fixtures/upstream/evaluate_activitynet_qa.py",
                                   "--commit", "abc123", "--source", str(src)])
    assert res.exit_code == 0, res.output
    s = yaml.safe_load(src.read_text())["t-vcg"]
    assert s["verified"] is True and s["upstream_commit"] == "abc123"


def test_verify_vcg_fails_on_difference(tmp_path):
    src = tmp_path / "source.yaml"
    shutil.copy("configs/prompts/judge/source.yaml", src)
    res = CliRunner().invoke(app, ["judge", "verify-vcg", "--upstream", "tests/fixtures/upstream/evaluate_msvd_qa_variant.py",
                                   "--commit", "abc123", "--source", str(src)])
    assert res.exit_code == 1 and "differs" in res.output
    assert yaml.safe_load(src.read_text())["t-vcg"]["verified"] is False


def test_vendor_rub_cli(tmp_path):
    src = tmp_path / "source.yaml"
    shutil.copy("configs/prompts/judge/source.yaml", src)
    out = tmp_path / "t_rub.txt"
    res = CliRunner().invoke(app, ["judge", "vendor-rub", "--prior", "tests/fixtures/upstream/prior_judge_base.j2",
                                   "--var-map", "q=question,gt=answer,gen=pred", "--commit", "def456",
                                   "--out", str(out), "--source", str(src)])
    assert res.exit_code == 0, res.output
    assert "{question}" in out.read_text()
    s = yaml.safe_load(src.read_text())["t-rub"]
    assert s["verified"] is True and s["upstream_commit"] == "def456"
