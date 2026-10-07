from typer.testing import CliRunner

from tests.report.test_summary import _fill
from vjaudit.cli import app


def test_report_summary_cli_writes_results(tmp_paths, tmp_path):
    _fill(tmp_paths.parent / "a" / "mac")
    out = tmp_path / "results_mac.md"
    res = CliRunner().invoke(app, ["report", "summary", "--profile", "mac", "--paths", str(tmp_paths),
                                   "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "profile: mac" in res.output and f"wrote {out}" in res.output
    assert "Agreement with `claude-manual`" in out.read_text()
