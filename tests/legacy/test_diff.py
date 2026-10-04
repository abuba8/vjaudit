from pathlib import Path

from vjaudit.legacy.diff import pairwise_identity
from vjaudit.legacy.load import load_legacy_cells, load_legacy_config, s1_one_shot_labels

FX = Path("tests/fixtures/legacy/legacy.yaml")


def test_load_cells_from_config():
    cells = load_legacy_cells(FX)
    assert len(cells) == 3
    assert cells[1].condition == "one" and cells[1].set == "S1" and cells[1].frames == 4


def test_pairwise_identity_on_fixture():
    df = pairwise_identity(load_legacy_cells(FX), split="test")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n"] == 4 and row["pct_identical"] == 75.0
    assert list(df.columns) == ["cond_a", "set_a", "cond_b", "set_b", "dataset", "frames", "n", "pct_identical"]


def test_s1_one_shot_label_found():
    cfg = load_legacy_config(FX)
    assert s1_one_shot_labels(Path(cfg["prompts_dir"])) == ["Yes"]
