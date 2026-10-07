import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text()
MAC_TARGETS = ["pin", "smoke", "run1", "run2", "judge-run1", "judge-run2", "annotate-export", "report"]


def test_makefile_has_mac_targets():
    for t in MAC_TARGETS:
        assert re.search(rf"^{re.escape(t)}:", MAKEFILE, re.M), t


def test_makefile_passes_profile():
    assert re.search(r"^export VJAUDIT_PROFILE := \$\(PROFILE\)$", MAKEFILE, re.M)


def test_ci_markers_exclude_mac():
    line = next(ln for ln in MAKEFILE.splitlines() if ln.startswith("CI_MARKERS"))
    assert "not mac" in line


def test_pyproject_mac_extra():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = py["project"]["optional-dependencies"]
    assert "transformers>=4.57" in extras["mac"] and any(d.startswith("torch>=2.5") for d in extras["mac"])
    vllm = next(d for d in extras["gpu"] if d.startswith("vllm"))
    assert "sys_platform == 'linux'" in vllm
    assert any(m.startswith("mac:") for m in py["tool"]["pytest"]["ini_options"]["markers"])


def _make_profile(env_extra: dict, *args) -> str:
    import os
    import shutil
    import subprocess

    import pytest
    if not shutil.which("make"):
        pytest.skip("make not installed")
    drop = ("PROFILE", "VJAUDIT_PROFILE", "MAKEFLAGS", "MAKELEVEL", "MFLAGS")
    env = {k: v for k, v in os.environ.items() if k not in drop} | env_extra
    out = subprocess.run(["make", "-s", "--eval=show-profile: ; @echo $(VJAUDIT_PROFILE)", "show-profile", *args],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_makefile_profile_precedence():
    assert _make_profile({}) == "server"
    assert _make_profile({"VJAUDIT_PROFILE": "mac"}) == "mac"            # the documented env var works with make
    assert _make_profile({"PROFILE": "mac"}) == "mac"
    assert _make_profile({"VJAUDIT_PROFILE": "mac"}, "PROFILE=server") == "server"  # explicit wins


def _make_dry(env_extra: dict, target: str, *args) -> str:
    import os
    import shutil
    import subprocess

    import pytest
    if not shutil.which("make"):
        pytest.skip("make not installed")
    drop = ("RUN", "CONDA_DEFAULT_ENV", "MAKEFLAGS", "MAKELEVEL", "MFLAGS")  # also when the suite runs under make
    env = {k: v for k, v in os.environ.items() if k not in drop} | env_extra
    return subprocess.run(["make", "--no-print-directory", "-n", target, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout


def test_makefile_uses_active_conda_env_instead_of_uv():
    out = _make_dry({"CONDA_DEFAULT_ENV": "dl"}, "sanity")
    assert out.lstrip().startswith("pytest -m") and "uv run" not in out
    assert "uv run" not in _make_dry({"CONDA_DEFAULT_ENV": "dl"}, "smoke")
    assert _make_dry({"CONDA_DEFAULT_ENV": "dl"}, "smoke").lstrip().startswith("python -m vjaudit.cli gen smoke")


def test_makefile_uses_uv_outside_conda_and_respects_override():
    import shutil

    import pytest
    if not shutil.which("uv"):
        pytest.skip("uv not installed")
    assert "uv run pytest" in _make_dry({}, "sanity")
    assert "uv run pytest" in _make_dry({"CONDA_DEFAULT_ENV": "base"}, "sanity")  # base env: still uv
    assert "uv run pytest" in _make_dry({"CONDA_DEFAULT_ENV": "dl"}, "sanity", "RUN=uv run")


def test_python_version_range_allows_311_to_313():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert py["project"]["requires-python"] == ">=3.11,<3.14"
