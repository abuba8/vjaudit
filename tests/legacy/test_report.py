from pathlib import Path

from vjaudit.legacy.report import write_sp0_report


def test_report_contains_identity_tables_tables_and_s1(tmp_path):
    out = tmp_path / "sp0_report.md"
    write_sp0_report(Path("tests/fixtures/legacy/legacy.yaml"), Path("tests/legacy/expected_tables.json"), out)
    text = out.read_text()
    assert "## Generation identity across conditions (test)" in text
    assert "## Generation identity across conditions (val)" in text
    assert "75.0" in text  # zero vs one-shot S1 on the fixture
    assert "S1 one-shot demonstration verdict label(s): Yes" in text
    assert "4-bit (per prior README)" in text
    assert "## Table 3 reproduction" in text and "## Table 2 reproduction" in text
