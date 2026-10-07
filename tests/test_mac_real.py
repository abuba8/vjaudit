"""Real-model checks for the Mac profile (spec §11). Never run by `make sanity`.

On the Mac: uv run pytest -m mac -v   (downloads Qwen3-VL-2B and Qwen2.5-1.5B on first run)
"""
from types import SimpleNamespace

import pytest

from tests.helpers.factories import make_item
from tests.test_e2e import _video
from vjaudit.config import GeneratorsFile, JudgesFile, load_yaml

pytestmark = pytest.mark.mac
MAC = "configs/profiles/mac"


def test_real_generator_answers(tmp_path):
    from vjaudit.data.frames import FrameCache
    from vjaudit.generate.factory import make_engine
    from vjaudit.generate.smoke import smoke_generators

    gen = next(g for g in load_yaml(f"{MAC}/generators.yaml", GeneratorsFile).generators if g.id == "qwen3vl2b")
    _video(tmp_path / "v.avi", n=40)
    item = make_item(0, video_path=str(tmp_path / "v.avi"), video_id="v", question="what color is the screen?")
    rows = smoke_generators(SimpleNamespace(generators=[gen]), [item], FrameCache(tmp_path / "f", max_side=448),
                            engine_factory=lambda g: make_engine(g, allow_unpinned=True), n_items=1)
    assert [r["mode"] for r in rows] == ["video", "blind"], rows
    assert all(r["answer"] and not r["answer"].startswith("ERROR:") for r in rows), rows


def test_real_judge_reads_margin():
    from vjaudit.judge.factory import make_client
    from vjaudit.judge.smoke import smoke_judges

    cfg = next(j for j in load_yaml(f"{MAC}/judges.yaml", JudgesFile).judges if j.id == "qwen25-1.5b")
    rows = smoke_judges(SimpleNamespace(judges=[cfg]),
                        client_factory=lambda c, d: make_client(c, d, allow_unpinned=True))
    rub = next(r for r in rows if r["template"] == "t-rub")
    assert rub["margin"] is not None, rows
