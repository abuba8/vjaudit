from pathlib import Path

import yaml
from typer.testing import CliRunner

from vjaudit.cli import app
from vjaudit.store import ArtifactStore

RAW = Path("tests/fixtures/raw").resolve()


def test_check_availability_cli_stores_item_pool(tmp_path):
    paths = {"msvd_raw": str(RAW / "msvd"), "msvd_videos": str(tmp_path / "v"),
             "anetqa_raw": str(RAW / "anetqa"), "anetqa_videos": str(tmp_path / "v"),
             "nextqa_raw": str(RAW / "nextqa"), "nextqa_videos": str(tmp_path / "v"),
             "artifacts": str(tmp_path / "artifacts")}
    (tmp_path / "v").mkdir()
    pf = tmp_path / "paths.yaml"
    pf.write_text(yaml.safe_dump(paths))
    res = CliRunner().invoke(app, ["data", "check-availability", "--paths", str(pf),
                                   "--report", str(tmp_path / "avail.md")])
    assert res.exit_code == 0, res.output
    pool = list(ArtifactStore(tmp_path / "artifacts").iter("item_pool"))
    assert len(pool) == 6 + 3 + 3
    assert not any(r["available"] for r in pool)


def test_check_availability_warns_when_pool_smaller_than_main_sample(tmp_path):
    paths = {k: str(RAW / v) for k, v in {"msvd_raw": "msvd", "anetqa_raw": "anetqa", "nextqa_raw": "nextqa"}.items()}
    paths.update({"msvd_videos": str(tmp_path), "anetqa_videos": str(tmp_path), "nextqa_videos": str(tmp_path),
                  "artifacts": str(tmp_path / "a")})
    pf = tmp_path / "paths.yaml"
    pf.write_text(yaml.safe_dump(paths))
    res = CliRunner().invoke(app, ["data", "check-availability", "--paths", str(pf), "--report", str(tmp_path / "r.md")])
    assert "WARNING: anetqa has 0 available items < main sample 2000" in res.output


def test_sample_cli_writes_items_and_report(tmp_path):
    from tests.helpers.factories import make_item
    store = ArtifactStore(tmp_path / "a")
    for n in range(60):
        it = make_item(n, item_id=f"anetqa:test:{n}", benchmark="anetqa", qtype=["color", "motion"][n % 2])
        store.put("item_pool", it, it.item_id)
    grid = tmp_path / "g.yaml"
    grid.write_text(yaml.safe_dump({"name": "t", "seed": 42, "benchmarks": ["anetqa"],
                                    "sample_sizes": {"main": 40, "annotated": 20, "vid": 10},
                                    "msvd_min_per": 15, "frac_dev": 0.4}))
    pf = tmp_path / "paths.yaml"
    pf.write_text(yaml.safe_dump({"artifacts": str(tmp_path / "a")}))
    res = CliRunner().invoke(app, ["data", "sample", "--paths", str(pf), "--grid", str(grid),
                                   "--report", str(tmp_path / "samples.md")])
    assert res.exit_code == 0, res.output
    items = list(ArtifactStore(tmp_path / "a").iter("item"))
    assert len(items) == 60
    assert sum(r["in_main"] for r in items) == 40 and sum(r["human_split"] == "dev" for r in items) == 8
    assert "| anetqa | 60 | 40 | 20 | 10 | 8 | 12 |" in (tmp_path / "samples.md").read_text()
