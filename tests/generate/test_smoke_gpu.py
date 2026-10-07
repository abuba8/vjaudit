"""GPU smoke test (server only): each pinned generator answers 5 items, with video and blind."""
import numpy as np
import pytest

from tests.helpers.factories import make_item
from vjaudit.config import GeneratorsFile, load_yaml

pytestmark = pytest.mark.gpu


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _cuda(), reason="no CUDA")
@pytest.mark.parametrize("gen_id", ["qwen25vl7b", "qwen3vl8b", "qwen3vl2b", "internvl35_8b", "llavanextvideo7b"])
def test_generator_answers_video_and_blind(gen_id):
    from vjaudit.generate.adapters import HFChatRenderer, get_adapter
    from vjaudit.generate.engine import VllmEngine
    from vjaudit.generate.systems import SystemSpec

    gen = next(g for g in load_yaml("configs/profiles/server/generators.yaml", GeneratorsFile).generators if g.id == gen_id)
    adapter = get_adapter(gen.family, HFChatRenderer(gen.hf_repo, gen.revision, gen.family))
    engine = VllmEngine(gen)
    n = gen.frames_for("anetqa")
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (n, 448, 448, 3), dtype=np.uint8)
    items = [make_item(i, question=q) for i, q in enumerate(
        ["what is shown?", "who is there?", "what color dominates?", "is it daytime?", "how many objects?"])]
    for blind in (False, True):
        spec = SystemSpec("smoke", gen.id, "short", blind, 0 if blind else n)
        reqs = [adapter.build_request(it, None if blind else frames, spec) for it in items]
        outs = engine.generate(reqs, 32)
        assert all(text.strip() for text, _ in outs), outs
