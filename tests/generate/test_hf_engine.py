import types

import numpy as np
import pytest

from vjaudit.config import GeneratorsFile, load_yaml
from vjaudit.device import default_dtype, pick_device
from vjaudit.generate.hf_engine import TransformersEngine

MAC_GENS = load_yaml("configs/profiles/mac/generators.yaml", GeneratorsFile).generators
QWEN = MAC_GENS[0]
PINNED = QWEN.model_copy(update={"revision": "abc123"})
FRAMES = np.zeros((8, 4, 4, 3), np.uint8)


class FakeInputs(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.moved_to = None

    def to(self, device, **kw):
        self.moved_to = device
        return self


class FakeProcessor:
    VOCAB = {7: " a dog", 8: " \n"}

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        return "<chat>" + str(messages)

    def __call__(self, text, videos=None, return_tensors=None, **kw):
        self.calls.append({"text": text, "videos": videos, "return_tensors": return_tensors, **kw})
        return FakeInputs(input_ids=np.array([[1, 2, 3]]))

    def batch_decode(self, seqs, skip_special_tokens):
        assert skip_special_tokens is True
        return ["".join(self.VOCAB[int(t)] for t in s) for s in seqs]


class FakeModel:
    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.kwargs = None

    def generate(self, **kw):
        if self.fail:
            raise self.fail
        self.kwargs = kw
        return np.array([[1, 2, 3, 7, 8]])


def fake_loader(proc=None, model=None, seen=None):
    def load(repo, revision, dtype, device):
        if seen is not None:
            seen.append((repo, revision, dtype, device))
        return proc or FakeProcessor(), model or FakeModel()
    return load


def test_pick_device_order():
    def torch_with(mps, cuda):
        return types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
                                     cuda=types.SimpleNamespace(is_available=lambda: cuda))
    assert pick_device(torch_with(True, True)) == "mps"
    assert pick_device(torch_with(False, True)) == "cuda"
    assert pick_device(torch_with(False, False)) == "cpu"


def test_default_dtype():
    assert {d: default_dtype(d) for d in ("mps", "cuda", "cpu")} == {"mps": "float16", "cuda": "bfloat16",
                                                                      "cpu": "float32"}


def test_engine_auto_dtype_uses_device_default_and_explicit_dtype_wins():
    seen = []
    TransformersEngine(PINNED, device="mps", loader=fake_loader(seen=seen))
    TransformersEngine(PINNED.model_copy(update={"dtype": "bfloat16"}), device="mps", loader=fake_loader(seen=seen))
    assert seen == [("Qwen/Qwen2.5-VL-3B-Instruct", "abc123", "float16", "mps"),
                    ("Qwen/Qwen2.5-VL-3B-Instruct", "abc123", "bfloat16", "mps")]


def test_engine_video_and_blind_calls():
    proc, model = FakeProcessor(), FakeModel()
    eng = TransformersEngine(PINNED, device="cpu", loader=fake_loader(proc, model))
    out = eng.generate([{"prompt": "P1", "multi_modal_data": {"video": FRAMES}}, {"prompt": "P2"}], max_new_tokens=5)
    assert out == [("a dog", 2), ("a dog", 2)]
    assert proc.calls[0]["text"] == ["P1"] and proc.calls[0]["videos"][0] is FRAMES
    assert proc.calls[1]["text"] == ["P2"] and proc.calls[1]["videos"] is None
    assert proc.calls[0]["return_tensors"] == "pt"
    assert model.kwargs["do_sample"] is False and model.kwargs["max_new_tokens"] == 5


def test_engine_render_uses_processor_chat_template():
    eng = TransformersEngine(PINNED, device="cpu", loader=fake_loader())
    assert eng.render([{"role": "user", "content": "hi"}]).startswith("<chat>")


def test_engine_refuses_unpinned():
    with pytest.raises(ValueError, match="UNPINNED"):
        TransformersEngine(QWEN, device="cpu", loader=fake_loader())
    TransformersEngine(QWEN, device="cpu", loader=fake_loader(), allow_unpinned=True)  # allowed when asked


def test_engine_oom_hint():
    eng = TransformersEngine(PINNED, device="mps", loader=fake_loader(model=FakeModel(RuntimeError("MPS backend out of memory"))))
    with pytest.raises(RuntimeError, match="reduce n_frames"):
        eng.generate([{"prompt": "P"}], 4)


def test_engine_load_error_names_repo():
    def broken(repo, revision, dtype, device):
        raise OSError("404 Client Error")
    with pytest.raises(RuntimeError, match=r"Qwen/Qwen2\.5-VL-3B-Instruct.*404"):
        TransformersEngine(PINNED, device="cpu", loader=broken)


def test_engine_close_drops_model():
    eng = TransformersEngine(PINNED, device="cpu", loader=fake_loader())
    eng.close()
    assert eng.model is None and eng.processor is None


def test_engine_greedy_ignores_model_generation_defaults():
    """Pure greedy like vLLM's temperature 0: no repetition penalty, no sampling fields from generation_config;
    frames are passed as already sampled (video only)."""
    proc, model = FakeProcessor(), FakeModel()
    eng = TransformersEngine(PINNED, device="cpu", loader=fake_loader(proc, model))
    eng.generate([{"prompt": "P", "multi_modal_data": {"video": FRAMES}}, {"prompt": "Q"}], 4)
    assert model.kwargs["repetition_penalty"] == 1.0 and model.kwargs["do_sample"] is False
    assert model.kwargs["temperature"] is None and model.kwargs["top_p"] is None and model.kwargs["top_k"] is None
    assert proc.calls[0]["do_sample_frames"] is False and "do_sample_frames" not in proc.calls[1]
