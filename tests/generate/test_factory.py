import numpy as np

from tests.generate.conftest import FakeFrameCache, FakeRenderer
from tests.generate.test_hf_engine import PINNED, fake_loader
from tests.helpers.factories import make_item
from vjaudit.generate import factory
from vjaudit.generate.adapters import HFChatRenderer, get_adapter
from vjaudit.generate.coverage import run_subset
from vjaudit.generate.engine import FakeEngine
from vjaudit.generate.hf_engine import TransformersEngine
from vjaudit.generate.smoke import smoke_generators
from vjaudit.generate.systems import SystemSpec
from vjaudit.store import ArtifactStore


def test_make_engine_selects_backend(monkeypatch):
    monkeypatch.setattr(factory, "TransformersEngine",
                        lambda g, allow_unpinned=False: ("hf", g.id, allow_unpinned))
    monkeypatch.setattr(factory, "VllmEngine", lambda g, allow_unpinned=False: ("vllm", g.id, allow_unpinned))
    assert factory.make_engine(PINNED, allow_unpinned=True) == ("hf", "qwen25vl3b", True)
    assert factory.make_engine(PINNED.model_copy(update={"backend": "vllm"})) == ("vllm", "qwen25vl3b", False)


def test_make_renderer_uses_engine_render_for_transformers(monkeypatch):
    eng = TransformersEngine(PINNED, device="cpu", loader=fake_loader())
    assert factory.make_renderer(PINNED, eng) == eng.render
    monkeypatch.setattr(factory, "HFChatRenderer", lambda repo, rev, fam: ("hfchat", repo, fam))
    assert factory.make_renderer(PINNED, FakeEngine(lambda r: "x")) == ("hfchat", PINNED.hf_repo, "qwen-vl")
    assert HFChatRenderer  # the default renderer for vLLM engines


class ClosingEngine(FakeEngine):
    def __init__(self, gid, log):
        super().__init__(lambda r: "x")
        self.gid, self.log = gid, log
        log.append(("open", gid))

    def close(self):
        self.log.append(("close", self.gid))


def test_run_subset_closes_engine_before_next_generator(tmp_path):
    from vjaudit.config import GeneratorsFile, load_yaml
    gens = load_yaml("configs/profiles/mac/generators.yaml", GeneratorsFile).generators
    items = [make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True) for i in range(2)]
    log = []
    run_subset(ArtifactStore(tmp_path), items, gens, "annotated", engine_factory=lambda g: ClosingEngine(g.id, log),
               renderer_factory=lambda g, e: FakeRenderer(), frame_cache=FakeFrameCache())
    assert log == [("open", "qwen25vl3b"), ("close", "qwen25vl3b"), ("open", "qwen3vl2b"), ("close", "qwen3vl2b"),
                   ("open", "llavaov05b"), ("close", "llavaov05b")]


def test_llava_onevision_adapter_video_and_blind():
    spec = SystemSpec("s", "llavaov05b", "short", False, 4)
    frames = np.zeros((4, 8, 8, 3), np.uint8)
    req = get_adapter("llava-onevision", FakeRenderer()).build_request(make_item(1), frames, spec)
    assert '"type": "video"' in req["prompt"] and "what is 1?" in req["prompt"]
    assert req["multi_modal_data"]["video"] is frames
    blind = get_adapter("llava-onevision", FakeRenderer()).build_request(
        make_item(1), None, SystemSpec("s", "llavaov05b", "short", True, 0))
    assert "multi_modal_data" not in blind and '"type": "video"' not in blind["prompt"]


def test_smoke_generators_rows(tmp_path):
    from types import SimpleNamespace
    prof = SimpleNamespace(generators=[PINNED])
    items = [make_item(i) for i in range(3)]
    rows = smoke_generators(prof, items, FakeFrameCache(), engine_factory=lambda g: FakeEngine(lambda r: "a cat"),
                            renderer_factory=lambda g, e: FakeRenderer(), n_items=2)
    assert len(rows) == 4
    assert all(set(r) == {"generator", "item_id", "mode", "answer", "seconds", "device"} for r in rows)
    assert [r["mode"] for r in rows] == ["video", "blind", "video", "blind"] and rows[0]["answer"] == "a cat"


def test_smoke_generators_reports_a_failing_generator_and_continues(tmp_path):
    from types import SimpleNamespace
    second = PINNED.model_copy(update={"id": "second"})
    prof = SimpleNamespace(generators=[PINNED, second])

    def factory_(g):
        if g.id == "qwen25vl3b":
            raise RuntimeError("could not load Qwen/Qwen2.5-VL-3B-Instruct@abc123: boom")
        return FakeEngine(lambda r: "ok")
    rows = smoke_generators(prof, [make_item(0)], FakeFrameCache(), engine_factory=factory_,
                            renderer_factory=lambda g, e: FakeRenderer(), n_items=1)
    assert rows[0]["generator"] == "qwen25vl3b" and rows[0]["answer"].startswith("ERROR:") and "boom" in rows[0]["answer"]
    assert [r["answer"] for r in rows[1:]] == ["ok", "ok"]
