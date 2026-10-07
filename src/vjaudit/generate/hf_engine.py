"""Generation with Hugging Face transformers on one device (Apple GPU via MPS on the Mac profile).

One request at a time, greedy decoding. The processor that renders the chat template also prepares the
video tensors, so prompt and inputs always agree.
"""
from __future__ import annotations

import gc
from typing import Callable

from vjaudit.config import GeneratorCfg
from vjaudit.device import default_dtype, pick_device

OOM_HINT = "reduce n_frames or frame_max_side in configs/profiles/mac"
Loader = Callable[[str, "str | None", str, str], tuple]


def _default_loader(repo: str, revision: str | None, dtype: str, device: str) -> tuple:
    import torch  # mac extra
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=revision, dtype=getattr(torch, dtype))
    model.to(device).eval()
    return processor, model


class TransformersEngine:
    """Greedy decoding with transformers; refuses an unpinned revision unless allowed."""

    def __init__(self, gen: GeneratorCfg, device: str | None = None, allow_unpinned: bool = False,
                 loader: Loader | None = None):
        if gen.revision == "UNPINNED" and not allow_unpinned:
            raise ValueError(f"{gen.id}: revision is UNPINNED; run `make pin PROFILE=<profile>` "
                             "(vjaudit.cli gen pin-revisions) first, or pass --allow-unpinned for a quick test")
        self.gen = gen
        self.device = device or pick_device()
        self.dtype = default_dtype(self.device) if gen.dtype == "auto" else gen.dtype
        revision = None if gen.revision == "UNPINNED" else gen.revision
        try:
            self.processor, self.model = (loader or _default_loader)(gen.hf_repo, revision, self.dtype, self.device)
        except Exception as e:  # noqa: BLE001 — name the model whatever the hub/loader raised
            raise RuntimeError(f"could not load {gen.hf_repo}@{gen.revision}: {e}") from e

    def render(self, messages: list[dict]) -> str:
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _inputs(self, req: dict):
        video = req.get("multi_modal_data", {}).get("video")
        if video is None:
            inputs = self.processor(text=[req["prompt"]], videos=None, return_tensors="pt")
        else:  # frames were already sampled by the frame cache: do not let the processor re-sample them
            inputs = self.processor(text=[req["prompt"]], videos=[video], return_tensors="pt", do_sample_frames=False)
        model_dtype = getattr(self.model, "dtype", None)
        return inputs.to(self.device, dtype=model_dtype) if model_dtype is not None else inputs.to(self.device)

    def generate(self, requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]:
        out = []
        for req in requests:
            try:
                inputs = self._inputs(req)
                seq = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            except Exception as e:  # noqa: BLE001
                if "out of memory" in str(e).lower():
                    raise RuntimeError(f"{self.gen.id}: {e} — {OOM_HINT}") from e
                raise
            new = seq[0][inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode([new], skip_special_tokens=True)[0].strip()
            out.append((text, len(new)))
        return out

    def close(self) -> None:
        """Drop the model and free GPU memory before the next generator loads."""
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
