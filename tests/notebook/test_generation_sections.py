"""Sections 7–8: frames and generation, pinned to the package where the package has the same logic."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.factories import make_item
from tests.test_e2e import _video
from vjaudit.config import GeneratorsFile, load_yaml
from vjaudit.data.frames import FrameCache, sample_indices
from vjaudit.generate.adapters import fill_prompt
from vjaudit.generate.systems import expand_systems, parse_mc_letter

ROOT = Path(__file__).resolve().parents[2]
SERVER_GENS = load_yaml(ROOT / "configs/profiles/server/generators.yaml", GeneratorsFile).generators


def test_frame_indices_match_package(nb_lib):
    for args in [(100, 8), (7, 8), (300, 16, 10, 200), (33, 1), (1000, 32, 0, None)]:
        assert nb_lib["sample_indices"](*args) == sample_indices(*args)


def test_frame_cache_matches_package(nb_lib, tmp_path):
    _video(tmp_path / "v.avi", n=20)
    ours, oi = nb_lib["FrameCache"](tmp_path / "a", 16).get(tmp_path / "v.avi", "v", 8)
    theirs, ti = FrameCache(tmp_path / "b", 16).get(tmp_path / "v.avi", "v", 8, None, None)
    assert oi == ti and np.array_equal(ours, theirs)
    again, _ = nb_lib["FrameCache"](tmp_path / "a", 16).get(tmp_path / "missing.avi", "v", 8)  # served from cache
    assert np.array_equal(again, ours)


def test_systems_match_package(nb_lib):
    server = nb_lib["SERVER"]["generators"]
    for bench in ("msvd", "anetqa", "nextoe"):
        ours = [(s["system_id"], s["style"], s["blind"], s["n_frames"]) for g in server
                for s in nb_lib["expand_systems"](g, bench)]
        theirs = [(s.system_id, s.style, s.blind, s.n_frames) for s in expand_systems(SERVER_GENS, bench)]
        assert ours == theirs


def test_parse_mc_matches_package(nb_lib):
    for t in ["B", "(C)", "b", "Answer: D.", "E) cat", "none", "", "A.", "AB", "The answer is A"]:
        assert nb_lib["parse_mc_letter"](t) == parse_mc_letter(t), t


def test_prompt_text_matches_package(nb_lib):
    item = make_item(1, question="why {x}?", mc_options=["a {0}", "b", "c", "d", "e"], mc_answer_idx=0)
    for style in ("short", "sentence", "mc"):
        assert nb_lib["prompt_text"](item.question, style, ROOT / "configs/prompts", item.mc_options) == \
            fill_prompt(item, style)


def test_build_messages(nb_lib):
    bm = nb_lib["build_messages"]
    assert bm("qwen-vl", "Q", False) == [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Q"}]}]
    assert bm("llava-onevision", "Q", True) == [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]
    assert bm("internvl", "Q", False) == [{"role": "user", "content": "<video>\nQ"}]
    assert bm("internvl", "Q", True) == [{"role": "user", "content": "Q"}]


def _setup(nb_lib, tmp_path, n=4):
    S = nb_lib["resolve_settings"](dict(PROFILE="mac", DRY_RUN=True, DATA_ROOT=str(tmp_path), RUN="annotated",
                                        RUN_LIMIT=None, RUN_STAGES=None), tmp_path / "repo")
    S["prompts"] = ROOT / "configs" / "prompts"
    items = pd.DataFrame([dict(make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True).model_dump(),
                               available=True, video_path="", problem="") for i in range(n)])
    return S, items, nb_lib["Store"](S["runs"])


class Frames:
    def get(self, video_path, video_id, n, t_start=None, t_end=None):
        return np.zeros((n, 4, 4, 3), np.uint8), list(range(n))


def test_generation_resumes_without_duplicates(nb_lib, tmp_path):
    S, items, store = _setup(nb_lib, tmp_path)
    calls = {"n": 0}

    class Crashing(nb_lib["FakeEngine"]):
        def generate(self, requests, max_new_tokens):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("kernel died")
            return super().generate(requests, max_new_tokens)

    with pytest.raises(RuntimeError, match="kernel died"):
        nb_lib["run_generation"](store, items, S, frame_cache=Frames(), engine_factory=lambda g, S: Crashing(g))
    first = len(store.load("generations"))
    assert first == 2
    n_systems = sum(len(nb_lib["expand_systems"](g, "msvd")) for g in S["generators"])  # 5
    new = nb_lib["run_generation"](nb_lib["Store"](S["runs"]), items, S, frame_cache=Frames())
    gens = nb_lib["Store"](S["runs"]).load("generations")
    assert new == 4 * n_systems - first and len(gens) == 4 * n_systems and gens["gen_id"].is_unique
    assert set(gens.columns) >= {"gen_id", "item_id", "benchmark", "system_id", "generator_id", "style", "blind",
                                 "n_frames", "frame_indices", "output_text", "answer_norm", "generator_rev"}


def test_run_limit_and_rerun_is_noop(nb_lib, tmp_path):
    S, items, store = _setup(nb_lib, tmp_path)
    S["run_limit"] = 2
    assert nb_lib["run_generation"](store, items, S, frame_cache=Frames()) == 2 * 5
    assert nb_lib["run_generation"](store, items, S, frame_cache=Frames()) == 0


def test_section_needs_previous_output_message(nb_lib, tmp_path):
    S, _, store = _setup(nb_lib, tmp_path)
    with pytest.raises(RuntimeError, match="run section 6"):
        nb_lib["run_generation"](store, pd.DataFrame(), S)


def test_undecodable_video_is_skipped_and_listed(nb_lib, tmp_path):
    S, items, store = _setup(nb_lib, tmp_path, n=2)

    class Broken(Frames):
        def get(self, video_path, video_id, n, t_start=None, t_end=None):
            if video_id == "vid0":
                raise nb_lib["FrameDecodeError"]("cannot decode vid0")
            return super().get(video_path, video_id, n)

    nb_lib["run_generation"](store, items, S, frame_cache=Broken())
    gens = store.load("generations")
    assert not ((gens.item_id == "msvd:test:0") & ~gens.blind).any()
    assert ((gens.item_id == "msvd:test:0") & gens.blind).any()
