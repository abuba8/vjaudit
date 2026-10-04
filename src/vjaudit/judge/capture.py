"""GPT-3.5 capture: stages in a fixed priority order → JudgeCall records (plan Task 13)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from vjaudit.config import GridCfg
from vjaudit.hashing import short_id
from vjaudit.schemas import Item, JudgeCall
from vjaudit.store import ArtifactStore, stamp

from .api_openai import BudgetExceeded, ModelRetired, OpenAIJudgeClient
from .parse import parse_vcg, parse_yes_no
from .readout import p_yes, readout_first_token, readout_vcg
from .templates import load_template, protocol_id, render_rub, render_vcg

INPUT_MODE = "text"


@dataclass(frozen=True)
class CaptureStage:
    name: str
    judge_id: str
    subset: Literal["annotated", "main"]
    systems: Literal["core", "ablation"]
    template: Literal["t-vcg", "t-rub"]
    decoding_tag: str
    request: dict

    @property
    def protocol_id(self) -> str:
        return protocol_id(load_template(self.template), self.decoding_tag)


def load_stages(grid: GridCfg) -> list[CaptureStage]:
    return [CaptureStage(**s) for s in grid.capture_stages]


@dataclass
class Job:
    stage: CaptureStage
    gen: dict
    item: Item
    judge_call_ids: list[str]  # one per sample (n)


@dataclass
class CaptureSummary:
    done: int = 0
    per_stage: dict = field(default_factory=dict)
    stopped_reason: str = ""
    spent_usd: float = 0.0


def judge_call_id(gen_id: str, judge_id: str, prot: str, sample_idx: int) -> str:
    return short_id(gen_id, judge_id, prot, "none", INPUT_MODE, str(sample_idx))


def build_jobs(store: ArtifactStore, stages: list[CaptureStage], items: list[Item],
               groups: dict[str, set[str]]) -> list[Job]:
    """Jobs in stage order, then gen_id; judge calls already stored or already scheduled are skipped."""
    by_id = {i.item_id: i for i in items}
    gens = sorted(store.iter("generation"), key=lambda g: g["gen_id"])
    scheduled: set[str] = set()
    jobs: list[Job] = []
    for st in stages:
        prot = st.protocol_id
        flag = "in_annotated" if st.subset == "annotated" else "in_main"
        for g in gens:
            it = by_id.get(g["item_id"])
            if it is None or not getattr(it, flag) or g["system_id"] not in groups[st.systems]:
                continue
            ids = [judge_call_id(g["gen_id"], st.judge_id, prot, k) for k in range(st.request.get("n", 1))]
            if all(store.has("judgecall", i) or i in scheduled for i in ids):
                continue
            scheduled.update(ids)
            jobs.append(Job(st, g, it, ids))
    return jobs


def _status(*candidates: str) -> str:
    for s in ("unparseable", "off_vocab", "api_no_logprob"):
        if s in candidates:
            return s
    return "ok"


def _records(job: Job, client: OpenAIJudgeClient, resp: dict) -> list[JudgeCall]:
    st = job.stage
    n = max(len(resp["choices"]), 1)
    out = []
    for k, ch in enumerate(resp["choices"]):
        text = (ch.get("message") or {}).get("content") or ""
        content = (ch.get("logprobs") or {}).get("content") or []
        if st.template == "t-vcg":
            verdict, score, pstat = parse_vcg(text)
            margin, top = readout_vcg(content)
        else:
            verdict, pstat = parse_yes_no(text)
            score = None
            margin, top = readout_first_token(content)
        status = _status(pstat, "api_no_logprob" if margin is None else "ok")
        decoding = {kk: v for kk, v in st.request.items() if v is not None}
        rec = dict(judge_call_id=job.judge_call_ids[k], gen_id=job.gen["gen_id"], judge_id=st.judge_id,
                   judge_rev=client.cfg.model, protocol_id=st.protocol_id, demo_set_id=None, input_mode=INPUT_MODE,
                   n_frames=0, margin=margin, p_yes=p_yes(margin) if margin is not None else None, top_tokens=top,
                   parsed_verdict=verdict, vcg_score=score, parse_status=status, latency_s=resp["_latency_s"],
                   cost_usd=resp["_cost_usd"] / n, sample_idx=k, decoding=decoding,
                   api_model_returned=resp.get("model"), system_fingerprint=resp.get("system_fingerprint"),
                   raw_text=text)
        out.append(JudgeCall(**stamp(rec, {"stage": st.name, "protocol_id": st.protocol_id, "judge": st.judge_id})))
    return out


async def _one(store: ArtifactStore, client: OpenAIJudgeClient, job: Job) -> int:
    st, it, g = job.stage, job.item, job.gen
    render = render_vcg if st.template == "t-vcg" else render_rub
    r = st.request
    resp = await client.complete(render(it.question, it.answers_ref[0], g["output_text"]), max_tokens=r["max_tokens"],
                                 temperature=r.get("temperature"), logprobs=r["logprobs"],
                                 top_logprobs=r["top_logprobs"], n=r.get("n", 1))
    return sum(store.put("judgecall", rec, rec.judge_call_id) for rec in _records(job, client, resp))


async def run_capture(store: ArtifactStore, clients: dict[str, OpenAIJudgeClient], jobs: list[Job],
                      chunk: int = 64, decisions_path: Path = Path("docs/decisions.md")) -> CaptureSummary:
    """Run jobs in order, ``chunk`` at a time; stop cleanly on ModelRetired / BudgetExceeded."""
    summary = CaptureSummary()
    for start in range(0, len(jobs), chunk):
        batch = jobs[start:start + chunk]
        results = await asyncio.gather(*[_one(store, clients[j.stage.judge_id], j) for j in batch],
                                       return_exceptions=True)
        stop = None
        for j, res in zip(batch, results):
            if isinstance(res, BaseException):
                stop = stop or res
                continue
            summary.done += res
            summary.per_stage[j.stage.name] = summary.per_stage.get(j.stage.name, 0) + res
        if stop is not None:
            summary.stopped_reason = f"{type(stop).__name__}: {stop}"
            if isinstance(stop, ModelRetired):
                judge = next((j.stage.judge_id for j in batch), "?")
                when = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with open(decisions_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## {when[:10]} — API judge retired\n- {judge} returned model_not_found at {when}; "
                            f"captured {summary.done} calls in this run.\n")
            if not isinstance(stop, (ModelRetired, BudgetExceeded)):
                raise stop
            break
    summary.spent_usd = sum(c.spent_usd for c in clients.values())
    return summary


def stage_scope(store: ArtifactStore, st: CaptureStage, items: list[Item], groups: dict[str, set[str]]) -> list[str]:
    """Every judge_call_id the stage is responsible for (done or not)."""
    by_id = {i.item_id: i for i in items}
    flag = "in_annotated" if st.subset == "annotated" else "in_main"
    prot = st.protocol_id
    out = []
    for g in store.iter("generation"):
        it = by_id.get(g["item_id"])
        if it is not None and getattr(it, flag) and g["system_id"] in groups[st.systems]:
            out += [judge_call_id(g["gen_id"], st.judge_id, prot, k) for k in range(st.request.get("n", 1))]
    return out


def write_capture_report(store: ArtifactStore, stages: list[CaptureStage], items: list[Item],
                         groups: dict[str, set[str]], out: Path) -> None:
    calls = {r["judge_call_id"]: r for r in store.iter("judgecall")}
    lines = ["# GPT-3.5 capture report", "",
             "| stage | judge | expected | done | ok rate | api_no_logprob rate | yes rate | cost USD | models | fingerprints |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for st in stages:
        scope = stage_scope(store, st, items, groups)
        done = [calls[i] for i in scope if i in calls]
        n = len(done)
        if n:
            ok = sum(r["parse_status"] == "ok" for r in done) / n
            nolp = sum(r["parse_status"] == "api_no_logprob" for r in done) / n
            yes = sum(r["parsed_verdict"] is True for r in done) / n
            rates = f"{ok:.3f} | {nolp:.3f} | {yes:.3f}"
        else:
            rates = "— | — | —"
        cost = sum(r["cost_usd"] for r in done)
        models = ", ".join(sorted({str(r["api_model_returned"]) for r in done})) or "—"
        fps = ", ".join(sorted({str(r["system_fingerprint"]) for r in done})) or "—"
        lines.append(f"| {st.name} | {st.judge_id} | {len(scope)} | {n} | {rates} | {cost:.4f} | {models} | {fps} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
