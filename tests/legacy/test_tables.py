import json
from pathlib import Path

import pytest

from vjaudit.legacy.load import load_legacy_cells, load_legacy_config
from vjaudit.legacy.tables import acceptance, table2, table3

FX = Path("tests/fixtures/legacy/legacy.yaml")
EXPECTED = json.loads(Path("tests/legacy/expected_tables.json").read_text())
LOCAL = Path("configs/local/legacy.yaml")


def test_acceptance_counts_unparseable_as_no():
    judge = {"0": {"Similarity": "Yes"}, "1": {"Similarity": "garbled"}, "2": {"Similarity": "No"}, "3": {"Similarity": " yes "}}
    assert acceptance(judge) == 50.0


def test_table3_shape_from_fixture():
    t3 = table3(load_legacy_cells(FX))
    assert t3["nextqa"]["one"]["4"]["S1"] == 50.0
    assert t3["nextqa"]["one"]["4"]["zero"] == 75.0


def test_table2_fp_fn_from_fixture():
    cfg = load_legacy_config(FX)
    t2 = table2(load_legacy_cells(FX), cfg["consensus"])
    # judge Yes,Yes,No,No vs consensus Yes,No,Yes,No -> FP=1 (idx1), FN=1 (idx2)
    assert t2["nextqa"]["S2"] == {"J": 50.0, "FP": 1, "FN": 1}
    assert t2["nextqa"]["consensus"] == 50.0


@pytest.mark.legacy
@pytest.mark.skipif(not LOCAL.exists(), reason="configs/local/legacy.yaml not present")
def test_table3_reproduces_to_0p1():
    got = table3(load_legacy_cells(LOCAL))
    for ds, conds in EXPECTED["table3"].items():
        for cond, frames in conds.items():
            for fr, sets in frames.items():
                for s, v in sets.items():
                    assert got[ds][cond][fr][s] == pytest.approx(v, abs=0.1), (ds, cond, fr, s)


@pytest.mark.legacy
@pytest.mark.skipif(not LOCAL.exists(), reason="configs/local/legacy.yaml not present")
def test_table2_reproduces():
    cfg = load_legacy_config(LOCAL)
    got = table2(load_legacy_cells(LOCAL), cfg["consensus"])
    for ds, rows in EXPECTED["table2"].items():
        assert got[ds]["consensus"] == pytest.approx(rows["consensus"], abs=0.1)
        for s in ("S2", "S3", "S4", "S5"):
            assert got[ds][s]["J"] == pytest.approx(rows[s]["J"], abs=0.1)
            assert (got[ds][s]["FP"], got[ds][s]["FN"]) == (rows[s]["FP"], rows[s]["FN"])
