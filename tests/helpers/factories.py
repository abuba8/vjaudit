"""Factories that build valid records for tests."""
from vjaudit.schemas import Generation, Item, JudgeCall

STAMP = {"config_hash": "c" * 64, "git_sha": "deadbeef", "created_at": "2026-10-04T00:00:00+00:00"}


def make_item(i: int = 0, **kw) -> Item:
    base = dict(item_id=f"msvd:test:{i}", benchmark="msvd", split="test", video_id=f"vid{i}",
                video_path=f"/videos/vid{i}.avi", t_start=None, t_end=None, question=f"what is {i}?",
                answers_ref=["dog"], qtype="what", mc_options=None, mc_answer_idx=None,
                in_main=False, in_annotated=False, in_vid_subset=False, human_split=None,
                vid_rank=None, available=True, **STAMP)
    base.update(kw)
    return Item(**base)


def make_generation(i: int = 0, **kw) -> Generation:
    base = dict(gen_id=f"{i:016x}", item_id=f"msvd:test:{i}", system_id="qwen3vl8b",
                generator_repo="Qwen/Qwen3-VL-8B-Instruct", generator_rev="UNPINNED", style="short",
                blind=False, n_frames=16, frame_indices=list(range(16)),
                decoding={"temperature": 0, "max_new_tokens": 128}, output_text="a dog",
                answer_norm="dog", n_output_tokens=2, **STAMP)
    base.update(kw)
    return Generation(**base)


def make_judgecall(i: int = 0, **kw) -> JudgeCall:
    base = dict(judge_call_id=f"{i:016x}", gen_id=f"{i:016x}", judge_id="gpt-3.5-turbo-0125",
                judge_rev="gpt-3.5-turbo-0125", protocol_id="t-vcg@abcdef12/t0", demo_set_id=None,
                input_mode="text", n_frames=0, margin=1.2, p_yes=0.77, top_tokens=[["yes", -0.1]],
                parsed_verdict=True, vcg_score=4.0, parse_status="ok", latency_s=0.5, cost_usd=0.0001,
                sample_idx=0, decoding={"temperature": 0}, api_model_returned="gpt-3.5-turbo-0125",
                system_fingerprint=None, raw_text="{'pred': 'yes', 'score': 4}", **STAMP)
    base.update(kw)
    return JudgeCall(**base)
