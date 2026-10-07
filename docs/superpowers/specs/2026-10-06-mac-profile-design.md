# Mac Profile Design: Run the Pipeline Locally on an Apple-Silicon Laptop

**Date:** 2026-10-06 · **Status:** design approved in chat 2026-10-06; this spec awaits review.
**Parent specs:** `RESEARCH_DIRECTION.md` (locked), `REQUIREMENTS_CONTRACT.md`.
**Parent plan:** `docs/superpowers/plans/2026-10-04-stage1-capture-before-gpt35-shutdown.md` (already implemented on branch `stage1`).

## 1. Goal

Run the whole Stage-1 pipeline end to end on a 32 GB M5 MacBook, with no CUDA and no OpenAI key:

1. small video-language models answer questions;
2. small open judge models grade the answers;
3. Claude can label a subset by hand, through files exported for copy-in and imported back;
4. a report compares judges with that reference.

The GPU-server path stays intact behind a profile switch.

**Success criteria**

- `PROFILE=mac` runs every step on the Mac: data check, sampling, generation, judging, Claude export/import, report.
- Run 1 is 100 questions per benchmark. Run 2 is 1,000 per benchmark. Both are resumable.
- `make sanity` stays green. All new logic has CPU tests.
- `docs/EXPLAINED.md` explains the idea and the code step by step, in plain language.

**Non-goals (YAGNI)**

- MLX or any quantized backend.
- SmolVLM and other model families beyond the three listed.
- VLM judges that see frames. That is SP3, later.
- Changing the GPU-server behaviour beyond moving its configs into a profile folder.

## 2. Profiles

- A profile is a folder `configs/profiles/<name>/` holding three files:
  - `generators.yaml`
  - `judges.yaml`
  - `grid.yaml` — seed, sample sizes, frame settings, capture stages
- There are two profiles:
  - **`server`** — the current files `configs/generators.yaml`, `configs/judges.yaml` and `configs/grids/stage1.yaml`, moved here unchanged.
  - **`mac`**.
- The active profile comes from the environment variable `VJAUDIT_PROFILE`, default `server`. The Makefile passes `PROFILE` into it (`make <target> PROFILE=mac`). Every CLI command reads its config through `vjaudit.profile.load_profile()`.
- Artifacts go to `<paths.artifacts>/<profile>/`, so the two profiles never mix records. The frame cache and API/local-judge cache are also per profile: `<paths.frame_cache>/<profile>/` and `<paths.api_cache>/<profile>/`.
- `GridCfg` gains `frame_max_side: int = 768` (mac: 448). The frame cache uses it.

## 3. Mac generation backend

**`TransformersEngine(gen: GeneratorCfg, device: str | None = None)`** is selected when `GeneratorCfg.backend == "hf"`.

- **Device:** `mps` if available, else `cuda`, else `cpu`.
- **dtype:** float16 on MPS, bfloat16 on CUDA, float32 on CPU.
- **Loading:** `AutoProcessor.from_pretrained(repo, revision)` and `AutoModelForImageTextToText.from_pretrained(repo, revision, torch_dtype)`, then moved to the device. An `UNPINNED` revision is refused unless `allow_unpinned=True`, the same rule as `VllmEngine`.
- **`generate(requests, max_new_tokens)`** handles one request at a time:
  1. `processor(text=[prompt], videos=[frames] if video else None, return_tensors="pt")` → device;
  2. `model.generate(**inputs, do_sample=False, max_new_tokens=…)`;
  3. decode only the new tokens with `skip_special_tokens=True`.
  It returns `(text, n_new_tokens)`.
- **Renderer:** the engine exposes `render(messages)`, using the processor's `apply_chat_template(..., add_generation_prompt=True, tokenize=False)`. The adapter uses this renderer, so prompt and inputs come from the same processor.
- **New family `llava-onevision`:** the adapter is identical to `qwen-vl` (a `{"type":"video"}` item plus a text item).
- **Errors:**
  - A missing model or download failure exits with the repo id and the HF error.
  - A CUDA/MPS out-of-memory raises with the hint "reduce n_frames or frame_max_side in configs/profiles/mac".

**Mac generators** (`configs/profiles/mac/generators.yaml`, all `backend: hf`, `n_frames: {default: 8}`):

| id | repo | family | styles | blind |
|---|---|---|---|---|
| `qwen25vl3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | qwen-vl | short, sentence | yes |
| `qwen3vl2b` | `Qwen/Qwen3-VL-2B-Instruct` | qwen-vl | short | no |
| `llavaov05b` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | llava-onevision | short | no |

That gives 5 systems: `qwen25vl3b`, `qwen25vl3b-sentence`, `qwen25vl3b-blind`, `qwen3vl2b`, `llavaov05b`. There are no frame-ablation systems on the Mac.

## 4. Local judges

- `JudgeCfg` gains:
  - `backend: Literal["openai","transformers"] = "openai"`
  - `revision: str = "UNPINNED"`
  - `chat_template_kwargs: dict = {}`
- For `kind: open`, `model` is the HF repo id. Prices and `max_usd` are 0, and `concurrency` is 1.
- **`LocalTransport(cfg)`** is an async callable with the same contract as the OpenAI transport. It takes a request dict and returns a response dict, so `OpenAIJudgeClient` (cache, retry and error handling), `capture`, the parsers, the readout and the report are reused unchanged.
  1. It runs in a worker thread (`asyncio.to_thread`).
  2. It renders the prompt with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)`.
  3. Decoding is greedy when `temperature` is 0. When `temperature` is absent or > 0, it samples (temperature 1.0 if absent) and repeats `n` times.
  4. It calls `generate(..., max_new_tokens=max_tokens, output_logits=True, return_dict_in_generate=True)`.
  5. The per-step raw logits go to the pure function `content_from_logits(gen_ids, step_logits, decode, k)`. This produces OpenAI-shaped `logprobs.content`: token = `decode([id])`, `log_softmax` logprob, and the top-`k` alternatives (`k = top_logprobs`).
  6. It returns:
     - `{"model": repo, "system_fingerprint": revision, "choices": [{"index", "message": {"role":"assistant","content": text}, "logprobs": {"content": [...]}}], "usage": {"prompt_tokens", "completion_tokens"}}`;
     - with exactly `n` choices.
- **Local errors:** any exception inside generation becomes `TransportError(422, "local_error", msg)`. That is non-retriable, so capture records it as failed and continues.
- **Mac judges** (`configs/profiles/mac/judges.yaml`):
  - `qwen3-4b`: `Qwen/Qwen3-4B-Instruct-2507`, the main judge;
  - `qwen25-1.5b`: `Qwen/Qwen2.5-1.5B-Instruct`, the small judge.
  - The `gpt-3.5-turbo-0125` entry is kept for optional use.
- **Capture stages** (`configs/profiles/mac/grid.yaml`; `core` = all 5 systems):

| Stage | Items | Judge | Template | Request |
|---|---|---|---|---|
| M1 | annotated (run 1) | qwen3-4b | t-vcg | temperature 0, max_tokens 64, logprobs, top 20 |
| M2 | annotated | qwen3-4b | t-rub | temperature 0, max_tokens 1, logprobs, top 20 |
| M3 | annotated | qwen25-1.5b | t-vcg | as M1 |
| M4 | annotated | qwen25-1.5b | t-rub | as M2 |
| M5 | main (run 2) | qwen3-4b | t-vcg | as M1 |
| M6 | main | qwen3-4b | t-rub | as M2 |
| G1, G2 | annotated | gpt-3.5-turbo-0125 | t-vcg, t-rub | as S1/S2; **optional** |

- `CaptureStage` gains `optional: bool = False`. Optional stages run only when named in `--stages`. The capture CLI builds `LocalTransport` for `backend: transformers` judges.
- The provenance guard (verified templates) applies unchanged. The Mac runbook either verifies the prompts or uses `--allow-unverified` for a smoke run, stating that the result is then not paper-grade.

## 5. Samples (mac grid)

- `sample_sizes: {main: 1000, annotated: 100, vid: 30}`
- `msvd_min_per: 15`, `frac_dev: 0.4`, `seed: 42`
- Run 1 = `--subset annotated`. Run 2 = `--subset main`; it automatically skips pairs that run 1 already did.

## 6. Claude-manual and human annotation export/import (`vjaudit.annotate`)

**Units (R-ANN-07).** `build_units(items, generations) -> list[Unit]`:

- one unit per `(item_id, answer_norm)`;
- `unit_id = short_id(item_id, answer_norm)`;
- `display_answer` = the `output_text` of the first generation by `gen_id`;
- `gen_ids` and `system_ids` are lists.

Units are stored as kind `unit`.

**Export.** `annotate export --set annotated --for {claude,human} [--batch-size 100]` writes `annotation/<profile>/<for>/`:

- **`--for claude`** writes `batch_NN.md`. Each file is self-contained, so you can paste or attach it as is:
  - a short task statement;
  - the labelling rules (contract §5.4, condensed);
  - the 9 error tags (E1–E9);
  - the exact output format: a single CSV code block with header `unit_id,label,error_tags,note`, one row per unit, `label ∈ {correct, incorrect}`, `error_tags` separated by `;`, and no other text;
  - then the questions, each with its reference answer(s) and its candidates as `unit_id | candidate answer`.
  Units are grouped by question, and a question is never split across batches (R-ANN-08).
- **`--for human`** writes `batch_NN.csv` with columns `unit_id, benchmark, question, reference_answers, candidate_answer, label, error_tags, note`. The label columns are empty.
- Both write `manifest.json`, listing batch → unit_ids.
- The export does not show annotators system identity, judge verdicts or hypotheses (R-ANN-02).

**Import.** `annotate import --source <src> --file <csv> [--file …]`:

- The file may be a raw Claude reply: an optional code fence is stripped.
- Each row is validated:
  - `unit_id` must be exported;
  - `label` must be `correct` or `incorrect`;
  - tags must be within E1–E9.
- Errors are listed with row numbers, and nothing is stored if any row is invalid.
- `--source` must be either `claude-manual[-suffix]` or `human:<annotator-id>`.
- Records are of kind `label`: `{label_id = short_id(unit_id, source), unit_id, source, task: "ref", label, error_tags, note, batch}`.
- Re-importing the same unit and source is skipped with a count. Corrections go in under a new source suffix (for example `claude-manual-v2`).
- Units of a batch that are still missing are reported.

**Rule.** Claude labels are a reference **judge**. Reports name the source and never call it "human".

## 7. Report (`report summary`)

Writes `docs/results_<profile>.md`.

**Run-1 tables (annotated set, per benchmark):** one row per system with:

- `n` (generations);
- accuracy under each judge config (judge × template; accuracy = parsed-verdict yes-rate; unparseable count shown);
- accuracy under each label source;
- for each judge config versus the first label source: gap (judge − reference, in points), Cohen's κ, FPR (judge yes when reference incorrect), FNR.

**Visual-gain line:** Δ = acc(`qwen25vl3b`) − acc(`qwen25vl3b-blind`) under every judge config and every label source. This is a first look at hypothesis HD.

**Run-2 table (main set):** accuracy per system per judge config.

All metrics are pure functions with hand-computed test values. Cohen's κ uses the standard formula; if the expected agreement is 1, it reports "undefined".

## 8. Smoke commands (run first on the Mac)

- `gen smoke`: each generator answers 2 items, with video and blind, and prints answer, seconds and device.
- `judge smoke`: each local judge makes one T-VCG and one T-RUB call on a fixed example, and prints the verdict, margin and top tokens.

## 9. Dependencies and tooling

- `pyproject.toml`:
  - new extra `mac` = `torch>=2.5`, `torchvision`, `transformers>=4.57`, `accelerate`, `huggingface_hub`;
  - `vllm` in the `gpu` extra gets the marker `sys_platform == 'linux'`;
  - if needed, `[tool.uv] conflicts` declares `gpu` and `mac` as mutually exclusive.
- Makefile targets accept `PROFILE`. New targets: `smoke`, `run1`, `run2`, `judge-run1`, `judge-run2`, `annotate-export`, `report`.

## 10. Documentation

- **`docs/EXPLAINED.md`** (new), in plain language:
  1. the research idea;
  2. how open-ended VideoQA is graded and where it goes wrong;
  3. what the pipeline does, step by step;
  4. each part of the code and why it exists;
  5. the Mac runbook, with exact commands and expected outputs;
  6. how to label with Claude;
  7. how to read the report;
  8. glossary.
- **`docs/GUIDE.md`**: profile section, updated config paths, and a link to `EXPLAINED.md`.

## 11. Testing

- **CPU tests with fakes:**
  - profile loading and per-profile artifact roots;
  - engine selection by backend;
  - `TransformersEngine` plumbing with a duck-typed fake processor and model;
  - `LocalTransport` with a fake tokenizer and model returning numpy logits;
  - `content_from_logits` exact values;
  - capture with local judges end to end;
  - optional stages skipped by default;
  - units/export/import (round trip, fence stripping, validation errors, source rule, no system identity in exports);
  - report metrics (κ, FPR/FNR, gap, visual gain) on hand-made data.
- **Mac-only (marker `mac`):** a real-model smoke test of one generator and one judge. It is not run in CI.

## 12. Risks

| Risk | Mitigation |
|---|---|
| A model's processor rejects raw frame arrays, or its chat template differs | `gen smoke` reveals it in minutes. Per-family adapters keep fixes local. If one model fails, drop it from `generators.yaml` and record it in `docs/decisions.md`. |
| MPS speed or memory | 8 frames at 448 px, batch size 1. Run 2 is resumable. |
| `transformers` version drift (Qwen3-VL needs ≥ 4.57) | Pinned in `uv.lock`. |
| Judge prompts unverified on the Mac | Verify with the existing scripts (git clone works on the Mac). Otherwise use `--allow-unverified` for testing only. |
