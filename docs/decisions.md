# Decisions log

Dated entries for every change to a MUST, a threshold, a schema, a model substitution,
or an integrity finding (contract §0 item 5).

## 2026-10-04 — Stage 1 setup
- Research direction B locked (RESEARCH_DIRECTION.md, 2026-10-03); venue ICCV 2027 full scope.
- Benchmarks T1: MSVD-QA, ActivityNet-QA, NExT-OE. MSRVTT-QA and TGIF-QA dropped from Stage 1.

## 2026-10-04 — Schema additions (Stage 1, plan Task 2)
- `Item.vid_rank: int | None` (stratified order of the video-labelled subset; any prefix ≥150 usable).
- `Item.available: bool` (video present and decodable; unavailable items are never sampled).
- `Generation.style` adds `"mc"` (NExT-QA multiple-choice anchor, G-ANCHOR).
- `JudgeCall.sample_idx`, `decoding`, `api_model_returned`, `system_fingerprint`, `raw_text`.
- `JudgeCall.vcg_score` is `float | None` (contract said int; the official T-VCG prompt's example emits 4.8).
- Every record carries `config_hash`, `git_sha`, `created_at` as schema fields (base class `Record`).

## 2026-10-04 — SP0 (R-SP0-02/03) — PENDING
- Tooling ready (`scripts/sp0_report.py`). Outcome (a) cache reuse / (b) exemplars never reached the
  generator / (c) other must be recorded here after running on the machine with the legacy outputs.
  If (b): Gate G0 — notify all authors; record their decision on a BlackboxNLP correction.

## 2026-10-06 — Mac profile (pipeline test on a 32 GB Apple-Silicon laptop)
- New profile `configs/profiles/mac/`: generators Qwen2.5-VL-3B (short, sentence, blind), Qwen3-VL-2B and
  LLaVA-OneVision-0.5B via `transformers` on MPS, 8 frames at 448 px; local judges Qwen3-4B-Instruct-2507
  (main) and Qwen2.5-1.5B-Instruct (small); stages M1–M6, GPT-3.5 stages G1/G2 optional.
- Samples: main 1,000 / annotated 100 / vid 30 per benchmark (Run 2 / Run 1).
- No OpenAI key on the Mac: reference labels come from Claude by hand (`claude-manual`), exported as
  `annotation/mac/claude/batch_NN.md`. They are a reference judge, never reported as human labels.
- The server configs moved unchanged to `configs/profiles/server/`; per-profile artifacts and reports.
- Spec: `docs/superpowers/specs/2026-10-06-mac-profile-design.md`; plan: `docs/superpowers/plans/2026-10-06-mac-profile.md`.
