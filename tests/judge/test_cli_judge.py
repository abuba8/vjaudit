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


def test_judge_smoke_cli_prints_rows_and_fails_on_error(tmp_paths, monkeypatch):
    from vjaudit.judge import smoke
    monkeypatch.setattr(smoke, "smoke_judges", lambda prof, client_factory=None: [
        {"judge": "qwen3-4b", "template": "t-rub", "verdict": True, "margin": 1.25, "top_tokens": [["Yes", -0.1]],
         "seconds": 0.5},
        {"judge": "qwen25-1.5b", "template": "t-rub", "verdict": "ERROR: no weights", "margin": None,
         "top_tokens": [], "seconds": 0.0}])
    res = CliRunner().invoke(app, ["judge", "smoke", "--profile", "mac", "--paths", str(tmp_paths)])
    assert "profile: mac" in res.output and "qwen3-4b" in res.output and "1.250" in res.output
    assert "ERROR: no weights" in res.output and res.exit_code == 1


def test_pin_revisions_cli_pins_generators_and_judges(tmp_path, monkeypatch):
    import shutil

    from vjaudit.generate import pin
    prof = tmp_path / "p" / "mine"
    shutil.copytree("configs/profiles/mac", prof)
    monkeypatch.setattr(pin, "hf_resolver", lambda repo: "sha-x")
    res = CliRunner().invoke(app, ["gen", "pin-revisions", "--profile", str(prof)])
    assert res.exit_code == 0, res.output
    assert "qwen25vl3b: sha-x" in res.output and "qwen3-4b: sha-x" in res.output
    assert "UNPINNED" not in (prof / "generators.yaml").read_text()
    assert "revision: sha-x" in (prof / "judges.yaml").read_text()
