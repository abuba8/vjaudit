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
