from pathlib import Path

import pytest
from pydantic import ValidationError

from vjaudit.config import (
    BenchmarksFile,
    GeneratorCfg,
    GeneratorsFile,
    GridCfg,
    JudgesFile,
    load_yaml,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generators_yaml_validates():
    gens = load_yaml(ROOT / "configs/generators.yaml", GeneratorsFile)
    ids = [g.id for g in gens.generators]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert {"qwen25vl7b", "qwen3vl8b", "qwen3vl2b", "internvl35_8b", "llavanextvideo7b"} == set(ids)


def test_duplicate_generator_ids_rejected():
    g = dict(id="a", family="qwen-vl", hf_repo="x/y", revision="UNPINNED", backend="vllm",
             n_frames={"default": 16}, max_model_len=16384, styles=["short"], blind=False,
             frame_ablation=[], quantization=None)
    with pytest.raises(ValidationError):
        GeneratorsFile(generators=[GeneratorCfg(**g), GeneratorCfg(**g)])


def test_judges_yaml_has_both_gpt35_snapshots():
    judges = load_yaml(ROOT / "configs/judges.yaml", JudgesFile)
    by_id = {j.id: j for j in judges.judges}
    assert by_id["gpt-3.5-turbo-0125"].max_usd == 100
    assert by_id["gpt-3.5-turbo-1106"].max_usd == 20
    assert all(j.kind == "api" for j in judges.judges)


def test_benchmarks_and_grid_yaml_validate():
    b = load_yaml(ROOT / "configs/benchmarks.yaml", BenchmarksFile)
    assert {x.id for x in b.benchmarks} == {"msvd", "anetqa", "nextoe"}
    g = load_yaml(ROOT / "configs/grids/stage1.yaml", GridCfg)
    assert g.seed == 42 and g.sample_sizes == {"main": 2000, "annotated": 300, "vid": 200}


def test_local_dir_is_gitignored():
    assert "configs/local/*.yaml" in (ROOT / ".gitignore").read_text().splitlines()


def test_source_and_test_data_packages_are_not_gitignored():
    import subprocess
    for p in ("src/vjaudit/data/sampling.py", "tests/data/test_sampling.py"):
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT)
        assert r.returncode == 1, f"{p} is gitignored"
    r = subprocess.run(["git", "check-ignore", "-q", "data/msvd/videos/x.avi"], cwd=ROOT)
    assert r.returncode == 0, "top-level data/ must stay ignored"
