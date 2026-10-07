from pathlib import Path

import pytest
import yaml

EXAMPLE_PATHS = Path(__file__).resolve().parents[1] / "configs" / "paths.example.yaml"


@pytest.fixture(autouse=True)
def _no_ambient_profile(monkeypatch):
    """Tests choose their profile explicitly; `export PROFILE=mac` in the shell must not leak in."""
    monkeypatch.delenv("VJAUDIT_PROFILE", raising=False)


@pytest.fixture
def tmp_paths(tmp_path) -> Path:
    """A paths.yaml in tmp_path: the example locations, but artifacts/caches under tmp_path/a."""
    p = yaml.safe_load(EXAMPLE_PATHS.read_text(encoding="utf-8"))
    p["artifacts"] = str(tmp_path / "a")
    p["frame_cache"] = str(tmp_path / "a" / "frames")
    p["api_cache"] = str(tmp_path / "a" / "api_cache")
    out = tmp_path / "paths.yaml"
    out.write_text(yaml.safe_dump(p), encoding="utf-8")
    return out
