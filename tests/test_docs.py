"""The docs must not drift from the code: every make target, CLI command and repo path they name exists."""
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vjaudit.cli import app

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["docs/EXPLAINED.md", "docs/GUIDE.md", "README.md"]
MAKE_TARGETS = set(re.findall(r"^([a-z0-9][a-z0-9-]*):", (ROOT / "Makefile").read_text(), re.M))


def _text(doc):
    return (ROOT / doc).read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", DOCS)
def test_make_targets_exist(doc):
    named = set(re.findall(r"(?:^|`)[ \t]*make ([a-z0-9][a-z0-9-]*)", _text(doc), re.M))  # commands, not prose
    assert named - MAKE_TARGETS == set()


@pytest.mark.parametrize("doc", DOCS)
def test_cli_commands_exist(doc):
    for group, cmd in sorted(set(re.findall(r"vjaudit\.cli ([a-z]+) ([a-z][a-z-]*)", _text(doc)))):
        res = CliRunner().invoke(app, [group, cmd, "--help"])
        assert res.exit_code == 0, f"{doc}: vjaudit.cli {group} {cmd}"


@pytest.mark.parametrize("doc", DOCS)
def test_repo_paths_exist(doc):
    paths = re.findall(r"(?<![\w/])((?:src/vjaudit|configs|scripts|tests)/[\w./*<>{}-]*[\w*>}])", _text(doc))
    concrete = [p for p in sorted(set(paths)) if not re.search(r"[*<>{}]", p) and not p.startswith("configs/local")]
    missing = [p for p in concrete if not (ROOT / p).exists()]
    assert missing == []
