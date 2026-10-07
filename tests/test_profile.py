import shutil

from typer.testing import CliRunner

from vjaudit.cli import app
from vjaudit.paths import CONFIGS
from vjaudit.profile import PROFILE_ENV, active_profile_name, load_profile


def test_server_profile_loads_moved_configs(tmp_paths):
    p = load_profile("server", tmp_paths)
    assert p.name == "server" and len(p.generators) == 5 and p.grid.sample_sizes["main"] == 2000
    assert p.artifacts == tmp_paths.parent / "a" / "server"
    assert p.frame_cache == tmp_paths.parent / "a" / "frames" / "server"
    assert p.api_cache == tmp_paths.parent / "a" / "api_cache" / "server"
    assert p.grid.frame_max_side == 768


def test_mac_profile_values(tmp_paths):
    p = load_profile("mac", tmp_paths)
    assert [g.id for g in p.generators] == ["qwen25vl3b", "qwen3vl2b", "llavaov05b"]
    assert all(g.backend == "hf" and g.frames_for("anetqa") == 8 for g in p.generators)
    assert p.grid.frame_max_side == 448 and p.grid.sample_sizes == {"main": 1000, "annotated": 100, "vid": 30}
    assert (p.grid.msvd_min_per, p.grid.frac_dev, p.grid.seed) == (15, 0.4, 42)
    j = p.judge("qwen3-4b")
    assert j.backend == "transformers" and j.model == "Qwen/Qwen3-4B-Instruct-2507" and j.max_usd == 0
    small = p.judge("qwen25-1.5b")
    assert small.model == "Qwen/Qwen2.5-1.5B-Instruct" and small.concurrency == 1 and small.kind == "open"
    assert p.judge("gpt-3.5-turbo-0125").backend == "openai"


def test_mac_stage_table(tmp_paths):
    stages = {s["name"]: s for s in load_profile("mac", tmp_paths).grid.capture_stages}
    assert list(stages) == ["M1", "M2", "M3", "M4", "M5", "M6", "G1", "G2"]
    assert [stages[n]["judge_id"] for n in ("M1", "M2", "M3", "M4", "M5", "M6")] == \
        ["qwen3-4b", "qwen3-4b", "qwen25-1.5b", "qwen25-1.5b", "qwen3-4b", "qwen3-4b"]
    assert [stages[n]["subset"] for n in ("M1", "M4", "M5", "G1")] == ["annotated", "annotated", "main", "annotated"]
    assert stages["M1"]["request"] == {"temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": 20, "n": 1}
    assert stages["M2"]["request"]["max_tokens"] == 1 and stages["M2"]["template"] == "t-rub"
    assert stages["G1"]["optional"] and stages["G2"]["optional"] and not stages["M1"].get("optional", False)


def test_env_var_selects_profile(monkeypatch, tmp_paths):
    monkeypatch.setenv(PROFILE_ENV, "mac")
    assert active_profile_name() == "mac"
    assert load_profile(None, tmp_paths).name == "mac"


def test_default_profile_is_server(monkeypatch, tmp_paths):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert load_profile(None, tmp_paths).name == "server"


def test_profiles_write_to_separate_artifact_dirs(tmp_paths):
    assert load_profile("mac", tmp_paths).artifacts != load_profile("server", tmp_paths).artifacts
    assert load_profile("mac", tmp_paths).store().root == tmp_paths.parent / "a" / "mac"


def test_profile_dir_argument(tmp_path, tmp_paths):
    d = tmp_path / "p" / "mytest"
    shutil.copytree(CONFIGS / "profiles" / "mac", d)
    p = load_profile(d, tmp_paths)
    assert p.name == "mytest" and p.root == d and p.artifacts == tmp_paths.parent / "a" / "mytest"


def test_unknown_profile_names_the_choices(tmp_paths):
    import pytest

    with pytest.raises(FileNotFoundError, match="mac"):
        load_profile("nope", tmp_paths)


def test_cli_prints_profile(tmp_paths):
    res = CliRunner().invoke(app, ["gen", "coverage", "--subset", "annotated", "--profile", "mac",
                                   "--paths", str(tmp_paths)])
    assert res.exit_code == 0, res.output
    assert res.output.startswith("profile: mac")
