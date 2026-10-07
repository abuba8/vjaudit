"""Record schemas (contract §8.2) plus the Stage-1 additions logged in docs/decisions.md."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Every stored record carries provenance (contract §8.2 preamble)."""

    model_config = ConfigDict(extra="forbid")
    config_hash: str
    git_sha: str
    created_at: str


class Item(Record):
    item_id: str  # "{benchmark}:{split}:{native_qid}"
    benchmark: str
    split: str
    video_id: str
    video_path: str  # local only, never released
    t_start: float | None = None
    t_end: float | None = None
    question: str
    answers_ref: list[str]
    qtype: str
    mc_options: list[str] | None = None
    mc_answer_idx: int | None = None
    in_main: bool = False
    in_annotated: bool = False
    in_vid_subset: bool = False
    human_split: Literal["dev", "test"] | None = None
    # Stage-1 additions
    vid_rank: int | None = None
    available: bool = True


class Generation(Record):
    gen_id: str
    item_id: str
    system_id: str
    generator_repo: str
    generator_rev: str
    style: Literal["short", "sentence", "mc"]
    blind: bool
    n_frames: int
    frame_indices: list[int]
    decoding: dict
    output_text: str
    answer_norm: str
    n_output_tokens: int


class JudgeCall(Record):
    judge_call_id: str
    gen_id: str
    judge_id: str
    judge_rev: str
    protocol_id: str
    demo_set_id: str | None = None
    input_mode: Literal["text", "frames", "frames_noref"]
    n_frames: int = 0
    margin: float | None = None
    p_yes: float | None = None
    top_tokens: list[list] = []
    parsed_verdict: bool | None = None
    vcg_score: float | None = None
    parse_status: Literal["ok", "off_vocab", "unparseable", "api_no_logprob"]
    latency_s: float
    cost_usd: float
    # Stage-1 additions
    sample_idx: int = 0
    decoding: dict = {}
    api_model_returned: str | None = None
    system_fingerprint: str | None = None
    raw_text: str | None = None


class Unit(Record):
    """One distinct (question, normalized answer) pair to label; several systems can share it (R-ANN-07)."""

    unit_id: str  # short_id(item_id, answer_norm)
    item_id: str
    benchmark: str
    answer_norm: str
    display_answer: str  # output_text of the generation with the smallest gen_id
    gen_ids: list[str]
    system_ids: list[str]  # never exported to annotators (R-ANN-02)


class Label(Record):
    """A correctness label for a unit from one source: a person (human:<id>) or a reference judge (claude-manual)."""

    label_id: str  # short_id(unit_id, source)
    unit_id: str
    source: str
    task: Literal["ref", "vid"]
    label: Literal["correct", "incorrect"]
    error_tags: list[str]
    note: str
    batch: str
