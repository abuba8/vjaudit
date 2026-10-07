"""Profiles: one folder of configs per machine type (configs/profiles/<name>/).

A profile bundles three files: generators.yaml, judges.yaml and grid.yaml. The active profile comes from
``--profile`` or the environment variable VJAUDIT_PROFILE (default "server"). Each profile writes to its own
artifact, frame-cache and API-cache folders, so records from different profiles never mix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import GeneratorCfg, GeneratorsFile, GridCfg, JudgeCfg, JudgesFile, load_yaml
from .paths import CONFIGS, DEFAULT_PATHS_FILE, load_paths
from .store import ArtifactStore

PROFILE_ENV = "VJAUDIT_PROFILE"
DEFAULT_PROFILE = "server"
PROFILES_DIR = CONFIGS / "profiles"


def active_profile_name() -> str:
    return os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE


@dataclass(frozen=True)
class Profile:
    name: str
    root: Path
    generators: list[GeneratorCfg]
    judges: list[JudgeCfg]
    grid: GridCfg
    artifacts: Path
    frame_cache: Path
    api_cache: Path

    def store(self) -> ArtifactStore:
        return ArtifactStore(self.artifacts)

    def judge(self, judge_id: str) -> JudgeCfg:
        for j in self.judges:
            if j.id == judge_id:
                return j
        raise KeyError(f"judge {judge_id!r} is not in {self.root / 'judges.yaml'}")


def profile_root(name_or_dir: str | Path | None = None) -> Path:
    """The profile's config folder, without reading machine paths (used by commands that only edit configs)."""
    if name_or_dir is None:
        name_or_dir = active_profile_name()
    if isinstance(name_or_dir, Path) or "/" in str(name_or_dir) or os.sep in str(name_or_dir):
        root = Path(name_or_dir)  # a folder path (tests, custom profiles)
        if not (root / "grid.yaml").is_file():
            raise FileNotFoundError(f"{root} is not a profile folder (needs generators.yaml, judges.yaml, grid.yaml)")
        return root
    root = PROFILES_DIR / str(name_or_dir)
    if not root.is_dir():
        known = sorted(p.name for p in PROFILES_DIR.iterdir() if p.is_dir())
        raise FileNotFoundError(f"profile {name_or_dir!r} not found; known profiles: {', '.join(known)}")
    return root


def load_profile(name_or_dir: str | Path | None = None, paths_file: Path | str = DEFAULT_PATHS_FILE) -> Profile:
    """Load a profile by name (configs/profiles/<name>) or by directory; artifacts go to <artifacts>/<name>."""
    root = profile_root(name_or_dir)
    name = root.name
    paths = load_paths(paths_file)
    artifacts_base = paths["artifacts"]
    return Profile(
        name=name,
        root=root,
        generators=load_yaml(root / "generators.yaml", GeneratorsFile).generators,
        judges=load_yaml(root / "judges.yaml", JudgesFile).judges,
        grid=load_yaml(root / "grid.yaml", GridCfg),
        artifacts=artifacts_base / name,
        frame_cache=paths.get("frame_cache", artifacts_base / "frames") / name,
        api_cache=paths.get("api_cache", artifacts_base / "api_cache") / name,
    )
