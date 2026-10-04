import pytest

from tests.generate.conftest import FakeFrameCache, FakeRenderer
from vjaudit.generate.adapters import get_adapter
from vjaudit.generate.engine import FakeEngine
from vjaudit.generate.runner import run_generation
from vjaudit.generate.systems import expand_systems
from vjaudit.normalize import normalize_answer
from vjaudit.store import ArtifactStore


def _setup(gens, sid="qwen3vl8b"):
    gen = next(g for g in gens if g.id == "qwen3vl8b")
    spec = next(s for s in expand_systems(gens, "msvd") if s.system_id == sid)
    return gen, spec, get_adapter(gen.family, FakeRenderer())


def test_runner_one_generation_per_item_system_and_resumes(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    calls = {"n": 0}

    def crashing(req):
        calls["n"] += 1
        if calls["n"] == 8:
            raise RuntimeError("GPU died")
        return "a dog"

    store = ArtifactStore(tmp_path)
    with pytest.raises(RuntimeError):
        run_generation(store, items, spec, FakeEngine(crashing), FakeFrameCache(), gen, adapter, batch_size=1)
    assert len(list(ArtifactStore(tmp_path).iter("generation"))) == 7
    calls["n"] = 100
    new = run_generation(ArtifactStore(tmp_path), items, spec, FakeEngine(crashing), FakeFrameCache(), gen,
                         adapter, batch_size=4)
    rows = list(ArtifactStore(tmp_path).iter("generation"))
    assert new == 13 and len(rows) == 20 and len({(r["item_id"], r["system_id"]) for r in rows}) == 20


def test_rerun_is_noop(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    store = ArtifactStore(tmp_path)
    assert run_generation(store, items, spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen, adapter) == 20
    assert run_generation(store, items, spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen, adapter) == 0


def test_empty_and_refusal_outputs_stored_verbatim(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    outs = iter(["", "Sorry, I cannot see the video."])
    store = ArtifactStore(tmp_path)
    run_generation(store, items[:2], spec, FakeEngine(lambda r: next(outs)), FakeFrameCache(), gen, adapter)
    texts = sorted(r["output_text"] for r in store.iter("generation"))
    assert texts == ["", "Sorry, I cannot see the video."]


def test_generation_record_fields(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    store = ArtifactStore(tmp_path)
    run_generation(store, items[:1], spec, FakeEngine(lambda r: "The Dog."), FakeFrameCache(), gen, adapter)
    r = next(store.iter("generation"))
    assert r["n_output_tokens"] == 2 and r["frame_indices"] == list(range(16)) and r["n_frames"] == 16
    assert r["answer_norm"] == normalize_answer("The Dog.") == "dog"
    assert r["decoding"] == {"temperature": 0, "max_new_tokens": 128}
    assert r["generator_repo"] == "Qwen/Qwen3-VL-8B-Instruct" and r["style"] == "short" and not r["blind"]


def test_blind_system_skips_frames(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens, "qwen3vl8b-blind")
    fc = FakeFrameCache()
    store = ArtifactStore(tmp_path)
    run_generation(store, items[:3], spec, FakeEngine(lambda r: "x"), fc, gen, adapter)
    assert fc.calls == 0
    assert all(r["frame_indices"] == [] and r["n_frames"] == 0 and r["blind"] for r in store.iter("generation"))


def test_unavailable_items_are_skipped(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    items = [items[0].model_copy(update={"available": False})] + items[1:3]
    store = ArtifactStore(tmp_path)
    assert run_generation(store, items, spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen, adapter) == 2


def test_gen_id_depends_on_prompt_and_config(tmp_path, gens, items):
    gen, spec, adapter = _setup(gens)
    s1, s2 = ArtifactStore(tmp_path / "1"), ArtifactStore(tmp_path / "2")
    run_generation(s1, items[:1], spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen, adapter)
    gen2 = gen.model_copy(update={"revision": "abc123"})
    run_generation(s2, items[:1], spec, FakeEngine(lambda r: "x"), FakeFrameCache(), gen2, adapter)
    assert next(s1.iter("generation"))["gen_id"] != next(s2.iter("generation"))["gen_id"]


def test_undecodable_video_is_skipped_and_logged_not_fatal(tmp_path, gens, items, caplog):
    import logging

    from vjaudit.data.frames import FrameDecodeError

    gen, spec, adapter = _setup(gens)

    class BadOne(FakeFrameCache):
        def get(self, video_path, video_id, n, t_start, t_end):
            if video_id == "vid1":
                raise FrameDecodeError("vid1: cannot decode")
            return super().get(video_path, video_id, n, t_start, t_end)

    store = ArtifactStore(tmp_path)
    with caplog.at_level(logging.WARNING):
        new = run_generation(store, items[:3], spec, FakeEngine(lambda r: "x"), BadOne(), gen, adapter)
    assert new == 2 and "vid1" in caplog.text
    assert {r["item_id"] for r in store.iter("generation")} == {"msvd:test:0", "msvd:test:2"}
