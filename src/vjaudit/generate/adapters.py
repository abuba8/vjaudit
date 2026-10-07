"""Family-specific prompt/message construction for vLLM video inputs.

The chat template itself comes from the pinned HF processor/tokenizer (``HFChatRenderer``) on the
GPU server; tests inject a fake renderer. Prompt slots are filled with ``str.replace`` only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from vjaudit.paths import CONFIGS
from vjaudit.schemas import Item

from .systems import SystemSpec

Renderer = Callable[[list[dict]], str]


def load_prompt(style: str) -> str:
    return (CONFIGS / "prompts" / "generation" / f"{style}.txt").read_text(encoding="utf-8").rstrip("\n")


def fill_prompt(item: Item, style: str) -> str:
    text = load_prompt(style).replace("{question}", item.question)
    if style == "mc":
        if not item.mc_options:
            raise ValueError(f"{item.item_id} has no MC options")
        for k, opt in enumerate(item.mc_options):
            text = text.replace("{a%d}" % k, opt)
    return text


class Adapter:
    family = ""

    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    def messages(self, text: str, blind: bool) -> list[dict]:
        content = ([] if blind else [{"type": "video"}]) + [{"type": "text", "text": text}]
        return [{"role": "user", "content": content}]

    def build_request(self, item: Item, frames: np.ndarray | None, spec: SystemSpec) -> dict:
        req = {"prompt": self.renderer(self.messages(fill_prompt(item, spec.style), spec.blind))}
        if not spec.blind:
            if frames is None:
                raise ValueError(f"{spec.system_id} needs frames for {item.item_id}")
            req["multi_modal_data"] = {"video": frames}
        return req


class QwenVLAdapter(Adapter):
    family = "qwen-vl"


class LlavaNextVideoAdapter(Adapter):
    family = "llava-next-video"


class LlavaOneVisionAdapter(Adapter):
    """Same message structure as qwen-vl: a video item, then the text."""

    family = "llava-onevision"


class InternVLAdapter(Adapter):
    """InternVL chat templates take a plain string with a ``<video>`` placeholder."""

    family = "internvl"

    def messages(self, text: str, blind: bool) -> list[dict]:
        return [{"role": "user", "content": text if blind else f"<video>\n{text}"}]


ADAPTERS = {a.family: a for a in (QwenVLAdapter, LlavaNextVideoAdapter, LlavaOneVisionAdapter, InternVLAdapter)}


def get_adapter(family: str, renderer: Renderer) -> Adapter:
    return ADAPTERS[family](renderer)


class HFChatRenderer:
    """apply_chat_template from the pinned processor (qwen-vl, llava-next-video) or tokenizer (internvl)."""

    def __init__(self, hf_repo: str, revision: str, family: str):
        from transformers import AutoProcessor, AutoTokenizer  # gpu extra

        rev = None if revision == "UNPINNED" else revision
        if family == "internvl":
            self._tmpl = AutoTokenizer.from_pretrained(hf_repo, revision=rev, trust_remote_code=True)
        else:
            self._tmpl = AutoProcessor.from_pretrained(hf_repo, revision=rev)

    def __call__(self, messages: list[dict]) -> str:
        return self._tmpl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


GOLDEN_ITEM_QUESTION = "What is the man doing?"


def golden_render(family: str, mode: str) -> str:
    """Render the canonical golden prompt for ``family`` with the first pinned generator of that family."""
    from vjaudit.config import GeneratorsFile, load_yaml
    from vjaudit.store import stamp

    # Golden prompts belong to the server models, whatever profile is active.
    server = CONFIGS / "profiles" / "server" / "generators.yaml"
    gen = next(g for g in load_yaml(server, GeneratorsFile).generators if g.family == family)
    item = Item(**stamp(dict(item_id="golden:test:0", benchmark="golden", split="test", video_id="v",
                             video_path="", question=GOLDEN_ITEM_QUESTION, answers_ref=["x"], qtype="what"), {}))
    spec = SystemSpec("golden", gen.id, "short", mode == "blind", 0 if mode == "blind" else 4)
    frames = None if mode == "blind" else np.zeros((4, 8, 8, 3), np.uint8)
    return get_adapter(family, HFChatRenderer(gen.hf_repo, gen.revision, family)).build_request(item, frames, spec)["prompt"]
