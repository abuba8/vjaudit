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
    env = {k: v for k, v in os.environ.items() if k not in ("PROFILE", "VJAUDIT_PROFILE")} | env_extra
    out = subprocess.run(["make", "-s", "--eval=show-profile: ; @echo $(VJAUDIT_PROFILE)", "show-profile", *args],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_makefile_profile_precedence():
    assert _make_profile({}) == "server"
    assert _make_profile({"VJAUDIT_PROFILE": "mac"}) == "mac"            # the documented env var works with make
    assert _make_profile({"PROFILE": "mac"}) == "mac"
    assert _make_profile({"VJAUDIT_PROFILE": "mac"}, "PROFILE=server") == "server"  # explicit wins
