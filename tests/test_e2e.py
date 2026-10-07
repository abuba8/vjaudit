"""End-to-end on fixtures: loaders → availability → sampling → generation → GPT-3.5 capture → freeze."""
import shutil
from pathlib import Path

import av
import numpy as np

from tests.generate.conftest import FakeRenderer
from tests.judge.test_capture import Transport, jc
from vjaudit.cli_judge import _groups
from vjaudit.config import GeneratorsFile, GridCfg, load_yaml
from vjaudit.data.availability import check_availability
from vjaudit.data.frames import FrameCache
from vjaudit.data.loaders import load_benchmark
from vjaudit.data.sampling import assign_nested, split_dev_test
from vjaudit.freeze import freeze
from vjaudit.generate.coverage import coverage, run_subset
from vjaudit.generate.engine import FakeEngine
from vjaudit.generate.systems import expand_systems
from vjaudit.judge.api_openai import OpenAIJudgeClient
from vjaudit.judge.capture import build_jobs, load_stages, run_capture, write_capture_report
from vjaudit.schemas import Item
from vjaudit.store import ArtifactStore


def _video(path: Path, n=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for i in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, np.uint8), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


async def test_pipeline_end_to_end(tmp_path):
    vids = tmp_path / "videos"
    for stem in ("-abc_1_5", "xyz_10_20"):  # q-r_0_8 deliberately missing
        _video(vids / "YouTubeClips" / f"{stem}.avi")
    items = check_availability(load_benchmark("msvd", Path("tests/fixtures/raw/msvd")), vids,
                               report_path=tmp_path / "avail.md")
    assert sum(i.available for i in items) == 4  # 6 questions, 2 on the missing clip
    items = split_dev_test(assign_nested(items, sizes=(4, 3, 2), seed=42, msvd_min_per=0), 0.4, 42)
    store = ArtifactStore(tmp_path / "artifacts")
    for it in items:
        store.put("item", it, it.item_id)
    items = [Item(**r) for r in store.iter("item")]

    gens = load_yaml("configs/profiles/server/generators.yaml", GeneratorsFile).generators
    fc = FrameCache(tmp_path / "frames")
    n = run_subset(store, items, gens, "main", engine_factory=lambda g: FakeEngine(lambda r: "a dog"),
                   renderer_factory=lambda g, e: FakeRenderer(), frame_cache=fc)
    assert n == 4 * 11
    assert all(v == 0 for v in coverage(store, [i for i in items if i.in_main],
                                        [s.system_id for s in expand_systems(gens, "msvd")]).values())

    stages = load_stages(load_yaml("configs/profiles/server/grid.yaml", GridCfg))
    groups = _groups(gens)
    clients = {j: OpenAIJudgeClient(jc(j), tmp_path / "api_cache", transport=Transport())
               for j in ("gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106")}
    summary = await run_capture(store, clients, build_jobs(store, stages, items, groups),
                                decisions_path=tmp_path / "decisions.md")
    assert summary.stopped_reason == "" and summary.done > 0
    write_capture_report(store, stages, items, groups, tmp_path / "report.md")
    report = (tmp_path / "report.md").read_text()
    assert "| S1 |" in report and "| S7 |" in report
    assert build_jobs(store, stages, items, groups) == []  # everything captured; rerun is a no-op

    manifest = freeze(store, tmp_path / "api_cache", tmp_path / "freeze")
    assert {e.kind for e in manifest} == {"item", "generation", "judgecall", "api_cache"}
