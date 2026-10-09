import re

from tests.notebook.conftest import read_notebook, tags
from vjaudit.normalize import normalize_answer


def test_notebook_is_self_contained():
    for cell in read_notebook().cells:
        if cell.cell_type == "code":
            assert not re.search(r"^\s*(import vjaudit|from vjaudit)", cell.source, re.M), cell.source[:80]


def test_notebook_saved_without_outputs():
    for cell in read_notebook().cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None


def test_one_parameters_cell_and_toc_covers_sections():
    cells = read_notebook().cells
    assert sum("parameters" in tags(c) for c in cells) == 1
    headers = [line[3:].strip() for c in cells if c.cell_type == "markdown"
               for line in c.source.splitlines() if line.startswith("## ")]
    toc = next(c.source for c in cells if c.cell_type == "markdown" and "Table of contents" in c.source)
    for h in headers[1:]:
        assert h in toc, f"section '{h}' missing from the table of contents"


def test_every_section_explains_mac_vs_server():
    sections = "\n".join(c.source for c in read_notebook().cells if c.cell_type == "markdown").split("\n## ")[1:]
    for s in sections[3:12]:  # sections 3–11 describe a step (12 is troubleshooting)
        assert "Mac vs server" in s and "Reads" in s, s.splitlines()[0]


def test_normalize_matches_package(nb_lib):
    for text in ["A dog.", "  The Man is COOKING!! ", "an", "", "it's a cat-dog", "Ⅳ", "a  b\tc"]:
        assert nb_lib["normalize_answer"](text) == normalize_answer(text), text


def test_short_id_matches_package(nb_lib):
    from vjaudit.hashing import short_id
    assert nb_lib["short_id"]("a", "b", "c") == short_id("a", "b", "c")


def test_store_append_is_idempotent_and_load_dedups(nb_lib, tmp_path):
    st = nb_lib["Store"](tmp_path)
    assert st.append("t", [{"k": "1", "v": 1}, {"k": "2", "v": 2}], key="k") == 2
    assert st.append("t", [{"k": "2", "v": 9}, {"k": "3", "v": 3}], key="k") == 1
    df = st.load("t")
    assert list(df["k"]) == ["1", "2", "3"] and list(df["v"]) == [1, 2, 3]
    assert st.keys("t") == {"1", "2", "3"} and st.load("missing").empty


def test_need_names_the_section(nb_lib):
    import pandas as pd
    import pytest
    with pytest.raises(RuntimeError, match="run section 6"):
        nb_lib["need"](pd.DataFrame(), 6, "samples")


def test_resolve_settings_mac_and_dryrun(nb_lib, tmp_path):
    s = nb_lib["resolve_settings"](dict(PROFILE="mac", DRY_RUN=False, DATA_ROOT=None, RUN="annotated",
                                        RUN_LIMIT=None, RUN_STAGES=None), repo=tmp_path / "vjauditv2")
    assert s["data_root"] == tmp_path / "vjaudit" / "data" and s["runs"] == tmp_path / "vjauditv2" / "notebooks" / "runs" / "mac"
    assert [g["id"] for g in s["generators"]] == ["qwen25vl3b", "qwen3vl2b", "llavaov05b"]
    assert s["samples"] == {"main": 1000, "annotated": 100, "vid": 30} and s["frame_max_side"] == 448
    assert [st["name"] for st in s["stages"]] == ["M1", "M2", "M3", "M4"]
    d = nb_lib["resolve_settings"](dict(PROFILE="server", DRY_RUN=True, DATA_ROOT=str(tmp_path), RUN="main",
                                        RUN_LIMIT=None, RUN_STAGES=None), repo=tmp_path / "vjauditv2")
    assert d["samples"] == {"main": 3, "annotated": 3, "vid": 1} and d["runs"].name == "server-dryrun"
    assert all(g["backend"] == "fake" for g in d["generators"]) and all(j["backend"] == "fake" for j in d["judges"])
    assert [st["name"] for st in d["stages"]] == ["S3", "S4", "S6c", "S6d"]
