from typer.testing import CliRunner

from tests.annotate.helpers import gen, item
from vjaudit.cli import app
from vjaudit.store import ArtifactStore


def _fill(tmp_paths):
    store = ArtifactStore(tmp_paths.parent / "a" / "mac")
    for i in (1, 2):
        it = item(i)
        store.put("item", it, it.item_id)
    for g in (gen("a", "msvd:test:1", "qwen25vl3b", "dog", "dog"), gen("b", "msvd:test:2", "qwen3vl2b", "cat", "cat")):
        store.put("generation", __import__("vjaudit.schemas", fromlist=["Generation"]).Generation(**g), g["gen_id"])
    store.close()


def _run(*args):
    return CliRunner().invoke(app, list(args))


def test_export_import_status_round_trip(tmp_paths, tmp_path):
    _fill(tmp_paths)
    out = tmp_path / "ann"
    base = ["--profile", "mac", "--paths", str(tmp_paths)]
    res = _run("annotate", "export", "--set", "annotated", "--for", "claude", "--out-dir", str(out), *base)
    assert res.exit_code == 0, res.output
    md = (out / "mac" / "claude" / "batch_01.md").read_text()
    ids = [line.split("|")[1].strip() for line in md.splitlines() if line.startswith("| ") and "unit_id" not in line]
    assert len(ids) == 2 and len(list(ArtifactStore(tmp_paths.parent / "a" / "mac").iter("unit"))) == 2
    reply = tmp_path / "batch_01.txt"
    reply.write_text(f"Here you go:\n```csv\nunit_id,label,error_tags,note\n{ids[0]},correct,,\n```\n")
    res = _run("annotate", "import", "--source", "claude-manual", "--file", str(reply), "--out-dir", str(out), *base)
    assert res.exit_code == 0, res.output
    assert "stored 1" in res.output and "batch_01: 1 unit(s) still missing" in res.output
    res = _run("annotate", "status", "--out-dir", str(out), *base)
    assert "| batch_01 | 2 | claude-manual: 1 | 1 |" in res.output


def test_import_with_errors_stores_nothing_and_exits_1(tmp_paths, tmp_path):
    _fill(tmp_paths)
    out = tmp_path / "ann"
    base = ["--profile", "mac", "--paths", str(tmp_paths), "--out-dir", str(out)]
    _run("annotate", "export", "--set", "annotated", "--for", "claude", *base)
    bad = tmp_path / "bad.csv"
    bad.write_text("unit_id,label,error_tags,note\nnope,correct,,\n")
    res = _run("annotate", "import", "--source", "claude-manual", "--file", str(bad), *base)
    assert res.exit_code == 1 and "bad.csv row 1 (nope)" in res.output and "row bad.csv" not in res.output and "nothing was stored" in res.output
    assert list(ArtifactStore(tmp_paths.parent / "a" / "mac").iter("label")) == []


def test_import_rejects_bad_source(tmp_paths, tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("unit_id,label,error_tags,note\n")
    res = _run("annotate", "import", "--source", "claude", "--file", str(f), "--profile", "mac",
               "--paths", str(tmp_paths), "--out-dir", str(tmp_path / "ann"))
    assert res.exit_code == 2 and "claude-manual" in res.output
