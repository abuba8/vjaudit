import subprocess
from pathlib import Path

import pytest

SCRIPTS = ["download_msvd.sh", "download_anetqa.sh", "download_nextqa.sh", "download_all.sh", "fetch_upstream_prompts.sh"]
D = Path("scripts/download")


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_is_valid_bash(name):
    assert (D / name).exists()
    assert subprocess.run(["bash", "-n", str(D / name)]).returncode == 0


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_exits_zero_without_network(name):
    r = subprocess.run(["bash", str(D / name), "--help"], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and "Usage:" in r.stdout


def test_msvd_script_uses_official_sources():
    text = (D / "download_msvd.sh").read_text()
    assert "https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar" in text
    assert "mega.nz/#!QmxFwBTK!Cs7cByu_Qo42XJOsv0DjiEDMiEm8m69h60caDYnT_PQ" in text
    assert "mega.nz/#!QrowUADZ!oFfW_M5wAFsfuFDEJAIa2BeFVHYO0vxit3CMkHFOSfw" in text


def test_anet_and_next_scripts_use_expected_repos():
    assert "MILVLG/activitynet-qa" in (D / "download_anetqa.sh").read_text()
    assert "lmms-lab/ActivityNetQA" in (D / "download_anetqa.sh").read_text()
    assert "lmms-lab/NExTQA" in (D / "download_nextqa.sh").read_text()


def test_paths_example_matches_script_destinations():
    import yaml
    paths = yaml.safe_load(Path("configs/paths.example.yaml").read_text())
    all_scripts = "".join((D / s).read_text() for s in SCRIPTS)
    for key in ("msvd_raw", "msvd_videos", "anetqa_raw", "anetqa_videos", "nextqa_raw", "nextqa_videos"):
        assert paths[key] in all_scripts, key


def _run_python_with(tmp_path, env_extra):
    import os
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, says in (("python", "conda-python"), ("uv", "uv")):
        f = bin_dir / name
        f.write_text(f"#!/bin/sh\necho {says} \"$@\"\n")
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "CONDA_DEFAULT_ENV"}
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin", **env_extra)
    cmd = f'source "{D.resolve()}/common.sh"; run_python -m vjaudit.cli x'
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env).stdout.strip()


def test_run_python_prefers_the_active_conda_env(tmp_path):
    assert _run_python_with(tmp_path, {"CONDA_DEFAULT_ENV": "dl"}) == "conda-python -m vjaudit.cli x"
    assert _run_python_with(tmp_path, {}) == "uv run python -m vjaudit.cli x"
    assert _run_python_with(tmp_path, {"CONDA_DEFAULT_ENV": "base"}) == "uv run python -m vjaudit.cli x"


def test_prompt_script_runs_python_through_the_helper():
    text = (D / "fetch_upstream_prompts.sh").read_text()
    assert "run_python -m vjaudit.cli judge verify-vcg" in text and "uv run" not in text
