from pathlib import Path

import numpy as np
import pytest

from tests.generate.conftest import FakeRenderer
from tests.helpers.factories import make_item
from vjaudit.generate.adapters import get_adapter, load_prompt
from vjaudit.generate.systems import SystemSpec

VIDEO_TOKEN = {"qwen-vl": "video", "internvl": "<video>", "llava-next-video": "video"}
FRAMES = np.zeros((4, 8, 8, 3), np.uint8)


def spec(style="short", blind=False):
    return SystemSpec(system_id="x", generator_id="x", style=style, blind=blind, n_frames=0 if blind else 4)


@pytest.mark.parametrize("family", ["qwen-vl", "internvl", "llava-next-video"])
def test_video_request_has_video_and_question(family):
    req = get_adapter(family, FakeRenderer()).build_request(make_item(1), FRAMES, spec())
    assert VIDEO_TOKEN[family] in req["prompt"] and "what is 1?" in req["prompt"]
    assert req["multi_modal_data"]["video"] is FRAMES


@pytest.mark.parametrize("family", ["qwen-vl", "internvl", "llava-next-video"])
def test_blind_request_has_no_multimodal_data(family):
    req = get_adapter(family, FakeRenderer()).build_request(make_item(1), None, spec(blind=True))
    assert "multi_modal_data" not in req
    assert "<video>" not in req["prompt"] and '"type": "video"' not in req["prompt"]


def test_question_with_braces_renders_literally():
    item = make_item(1, question="What is {this}?")
    req = get_adapter("qwen-vl", FakeRenderer()).build_request(item, FRAMES, spec())
    assert "What is {this}?" in req["prompt"]
    assert "Answer the question using a single word or phrase." in req["prompt"]


def test_prompt_files_are_pinned():
    assert load_prompt("short") == "{question}\nAnswer the question using a single word or phrase."
    assert load_prompt("sentence") == ("{question}\nAnswer in one complete sentence, then briefly say what in "
                                       "the video supports your answer.")
    assert load_prompt("mc").endswith("E. {a4}\nAnswer with the option's letter from the given choices directly.")


def test_mc_prompt_fills_options_literally():
    item = make_item(1, question="why {x}?", mc_options=["a {0}", "b", "c", "d", "e"], mc_answer_idx=0)
    req = get_adapter("internvl", FakeRenderer()).build_request(item, FRAMES, spec(style="mc"))
    assert "A. a {0}" in req["prompt"] and "E. e" in req["prompt"] and "why {x}?" in req["prompt"]


GOLDEN = Path("tests/generate/golden")


@pytest.mark.hf
@pytest.mark.parametrize("family", ["qwen-vl", "internvl", "llava-next-video"])
@pytest.mark.parametrize("mode", ["video", "blind"])
def test_golden_prompt(family, mode):
    from vjaudit.generate.adapters import golden_render

    path = GOLDEN / f"{family}_{mode}.txt"
    if not path.exists():
        pytest.skip(f"{path} not rendered yet (run: uv run python -m vjaudit.cli gen render-golden)")
    assert golden_render(family, mode) == path.read_text(encoding="utf-8")
