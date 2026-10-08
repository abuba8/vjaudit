# Pipeline Notebook Design: the Whole Pipeline in One Self-Contained Jupyter Notebook

**Date:** 2026-10-08 · **Status:** design approved in chat 2026-10-08; this spec awaits review.
**Parent spec:** `docs/superpowers/specs/2026-10-06-mac-profile-design.md` (the Mac profile; its choices of
models, judges, samples and stages carry over unchanged).

## 1. Goal

A single notebook, `notebooks/pipeline.ipynb`, runs the whole pipeline from data to report. Its audience is
the project owner, who wants to read each step, run it, and see its result in one place.

- **Self-contained.** All pipeline logic is written in the notebook's cells. It never does
  `import vjaudit`. It reads only data files from the repository: the prompt texts in `configs/prompts/`
  and nothing else.
- **Explanatory.** Every section opens with markdown: what the step does, why, what it reads and writes,
  and what to change for the Mac or the server.
- **Runnable on both machines.** One settings cell switches between `PROFILE = "mac"` (32 GB Apple
  Silicon, transformers on MPS, local judges, Claude labels) and `PROFILE = "server"` (CUDA GPU, vLLM,
  GPT-3.5 through the OpenAI API).

**Success criteria**

- On the Mac, running all cells with `PROFILE = "mac"` produces answers, verdicts, Claude batch files,
  and, after labels are imported, a results report. This covers Run 1 (100 questions per benchmark) and
  Run 2 (1,000).
- Every long step can be stopped and re-run; finished work is skipped.
- `DRY_RUN = True` runs the notebook end to end in about a minute, with stand-in models and no downloads.
- CPU tests execute the notebook in dry-run mode on synthetic data, and check its core functions against
  the package's tested functions.

**Non-goals (YAGNI)**

- The SP0 integrity check, freeze/archive, golden-prompt rendering, and prompt-provenance verification.
  Those stay in the command-line version; verification keeps using `fetch_upstream_prompts.sh`.
- Multi-GPU sharding. The notebook runs one model at a time.
- Sharing results with the command-line version. The notebook keeps its own results folder.
- Changing `src/`, the Makefile or the existing tests.

## 2. Files

| File | Role |
|---|---|
| `notebooks/pipeline.ipynb` | the notebook |
| `notebooks/README.md` | three lines: what the notebook is, how to open it, link to section 0 |
| `.gitignore` | adds `/notebooks/runs/` |
| `pyproject.toml` | dev group gains `nbformat`, `nbclient`, `ipykernel` (for the notebook tests); a new `notebook` extra = `jupyterlab`, `ipykernel`, `tqdm`, `matplotlib` |
| `tests/notebook/` | notebook tests (section 9) |
| `README.md`, `WORKFLOW.md`, `docs/EXPLAINED.md`, `differences.md` | one short pointer each to the notebook |

## 3. Notebook structure

Each section is a markdown header, then explanation, then code cells. Section 0 holds a clickable table
of contents.

| § | Title | Content |
|---|---|---|
| 0 | How to use this notebook | Table of contents; Mac vs server; resuming; dry run; where results go |
| 1 | Prerequisites | **Commented-out** commands to install packages (`pip install …`) and to download the datasets (`bash ../scripts/download/download_all.sh`, or the per-benchmark scripts with `--annotations-only`); each with a one-line explanation. Nothing runs unless uncommented |
| 2 | Settings | The only cell to edit (tagged `parameters`). A markdown table lists every setting and its Mac and server values. Prints the resolved settings, device and package versions |
| 3 | Helpers | Results store (append-only JSON-lines tables, resumable), IDs, answer normalisation, progress bars |
| 4 | Load the benchmarks | MSVD-QA, ActivityNet-QA and NExT-OE into one table; shows rows and question-type counts |
| 5 | Check the videos | Finds each video and test-decodes it; table of available and missing per benchmark |
| 6 | Draw the samples | Nested main ⊃ annotated ⊃ vid, stratified by question type, seed 42, MSVD minimum per type, 40/60 dev/test; counts table |
| 7 | Frames | Evenly spaced frames, resized to `frame_max_side`, cached; shows one video's frames as a picture |
| 8 | Generate answers | Systems (short, sentence, blind, frame-ablation); prompt building; Mac engine (transformers) and server engine (vLLM); one model at a time; resumable; sample answers; optional NExT multiple-choice anchor (server) |
| 9 | Judge the answers | Templates from `configs/prompts/judge/`; local judge (transformers, top-20 log-probabilities from raw logits) and API judge (OpenAI, cache, retries, spend cap); verdict parsing; yes/no margin; stages in order; verdict table |
| 10 | Label with Claude | Units, batches, `batch_NN.md` export, validated CSV import, status table |
| 11 | Report | Accuracy per system under each judge and label source; gap, Cohen's κ, FPR, FNR; visual gain; writes `results.md` |
| 12 | Troubleshooting and next steps | The common failures and their fixes |

## 4. Settings (section 2)

One cell, tagged `parameters` (papermill convention), holding:

- `PROFILE = "mac"` (or `"server"`), `DRY_RUN = False`, `DATA_ROOT = None`.
  `DATA_ROOT = None` means `../../vjaudit/data` on the Mac (the data stays in v1's folder) and `../data` on
  the server.
- `RUN = "annotated"` (Run 1) or `"main"` (Run 2).
- `RUN_LIMIT = None`. A number caps the questions per benchmark in sections 8–9, for a first real test
  (for example 5).
- `RUN_STAGES = None`. None means every non-optional stage of the run; a list names stages explicitly,
  for example `["M1", "M2"]` or `["G1"]`.
- Two dictionaries, `MAC` and `SERVER`, written out in full in the cell: the notebook reads no YAML. The
  server values are copied from `configs/profiles/server/`. One dictionary is chosen by `PROFILE`.

| Setting | Mac | Server |
|---|---|---|
| generators | `qwen25vl3b` (Qwen/Qwen2.5-VL-3B-Instruct; short, sentence, blind), `qwen3vl2b` (Qwen/Qwen3-VL-2B-Instruct), `llavaov05b` (llava-hf/llava-onevision-qwen2-0.5b-ov-hf) | the five server generators from `configs/profiles/server/generators.yaml` (incl. blind, sentence, f1/f4 ablation) |
| generation backend | `transformers` on `mps` | `vllm` on `cuda` |
| frames | 8, max side 448 | 16 (ActivityNet 32), max side 768 |
| judges | `qwen3-4b` (Qwen/Qwen3-4B-Instruct-2507), `qwen25-1.5b` (Qwen/Qwen2.5-1.5B-Instruct), local | `gpt-3.5-turbo-0125` (cap 100 USD), `gpt-3.5-turbo-1106` (cap 20 USD), API |
| stages | M1–M6; G1, G2 optional | S1–S7 as in `configs/profiles/server/grid.yaml` |
| samples (main / annotated / vid) | 1,000 / 100 / 30 | 2,000 / 300 / 200 |

Shared by both: seed 42, `msvd_min_per` 15, `frac_dev` 0.4, generation prompts from
`configs/prompts/generation/`, max new tokens short 128 and sentence 256, and greedy decoding.

The table above is shown as markdown right above the cell. Each line in the cell also carries a comment
that names the other machine's value.

`DRY_RUN = True` overrides these settings:

- samples become 3 / 3 / 1;
- generators and judges become `fake` (deterministic stand-ins that load nothing);
- results go to `runs/<profile>-dryrun/`.

## 5. Results store (section 3)

- The folder is `notebooks/runs/<profile>/`, with `-dryrun` appended in a dry run.
- Tables: `pool.jsonl`, `items.jsonl`, `generations.jsonl`, `judgecalls.jsonl`, `units.jsonl`,
  `labels.jsonl`. Each row is one JSON object with a key column.
- `load(table)` returns a pandas DataFrame, keeping the first row per key.
- `append(table, rows)` writes only keys that are not stored yet, and flushes after each batch.
  A crash loses at most the batch in progress.
- `pool` and `items` are rebuilt only when `REBUILD = True` in their section. The old file is renamed with
  a timestamp first.
- IDs:
  - `gen_id = sha1(item_id | system_id | json(generation config incl. model, revision, frames, max side, prompt text, max tokens))[:16]`;
  - `judge_call_id = sha1(gen_id | judge_id | template_id@sha256(template text)[:8] | decoding_tag | sample_idx)[:16]`.

  So changing a model, a frame setting or a prompt creates new rows instead of reusing old ones.
- Caches:
  - frames: `runs/<profile>/frames/<video_id>_<n>_<t_start>_<t_end>_<max_side>.npz`;
  - judge replies: `runs/<profile>/judge_cache/<judge>/<sha256(request)>.json`.

## 6. Generation (section 8)

- **Systems.** For each generator:
  - one system per style (`<id>` for short, `<id>-sentence`);
  - `<id>-blind` (no video);
  - `<id>-f<k>` for each `frame_ablation` k.
- **Prompt.** Messages are a video item plus a text item for the Qwen-VL, LLaVA-OneVision and
  LLaVA-NeXT-Video families. InternVL uses the text `"<video>\n" + text` on the server. Blind requests
  drop the video item. The text comes from `configs/prompts/generation/<style>.txt`, with `{question}`
  replaced.
- **Mac engine.** `AutoProcessor` and `AutoModelForImageTextToText` at the pinned revision, with dtype
  float16 on `mps`, bfloat16 on `cuda` and float32 on `cpu`. The processor is called with
  `videos=[frames], do_sample_frames=False`. Generation passes
  `do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0`. Each request runs
  alone. The model is freed (`del`, `gc.collect()`, `torch.mps.empty_cache()`) before the next generator
  loads.
- **Server engine.** vLLM `LLM(model, revision, dtype, max_model_len, limit_mm_per_prompt={"video": 1})`
  with `SamplingParams(temperature=0)`, in batches of 16. The chat template comes from the HF processor
  (InternVL: its tokenizer with `trust_remote_code=True`).
- **Resume.** Before a generator loads, its missing `gen_id`s are computed. A generator with nothing
  missing is never loaded.
- **Errors.**
  - A video that cannot be decoded is skipped and listed.
  - Out-of-memory errors print the hint "lower frames or frame_max_side in the settings cell".
  - A model that fails to load is reported, and the loop continues with the next generator.
- **MC anchor (server, optional).** NExT questions with options, prompt `mc.txt`, letter parsing,
  exact-match accuracy table.

## 7. Judging (section 9)

- **Templates.** T-VCG is the system and user text from `t_vcg_system.txt` / `t_vcg_user.txt`; T-RUB is
  `t_rub.txt`. `{question}`, `{answer}` and `{pred}` are filled in one pass, so braces inside values stay
  literal. The reference used is the item's first reference answer.
- **Local judge.** `AutoTokenizer` and `AutoModelForCausalLM` at the pinned revision. The chat template is
  applied with `add_generation_prompt=True`. Greedy decoding uses the same overrides as generation, plus
  `output_logits=True, return_dict_in_generate=True`. For each generated step it records the token and the
  top 20 `(token, logprob)` pairs from `log_softmax` of the raw logits. Only one local judge model is
  loaded at a time; stages are grouped by judge.
- **API judge.** The `openai` client calls `chat.completions.create(model, messages, max_tokens, n,
  logprobs=True, top_logprobs=20, temperature?)`. On the server it uses:
  - a disk cache keyed by request;
  - retries with exponential backoff on 429 / 5xx / connection errors, up to 6 attempts;
  - `model_not_found` stops the stage;
  - a spend cap per judge, estimated from usage at $0.50 / $1.50 per 1M input/output tokens.
- **Reading a reply.**
  - T-VCG: try a dict literal; otherwise a regex for `pred: yes|no`.
  - T-RUB: the whole reply, stripped, must be `yes` or `no`.
  - Status is `unparseable` or `off_vocab` when it fails; it is never guessed.
  - The margin is logsumexp(yes spellings) − logsumexp(no spellings) at the verdict position. That is the
    token after `'pred': '` for T-VCG and the first token for T-RUB. Spellings ignore case, spaces, quotes
    and colons. If none is present, the margin is None and the status is `api_no_logprob`.
- **Stages.** Run in config order. Optional stages run only if listed in `RUN_STAGES`. Each stage covers
  its set's generations for its systems (`core` = all non-ablation systems; `ablation` = `-f<k>`). Calls
  already stored are skipped. A failed call is listed and does not stop the stage. Stage `n > 1` stores
  one row per sample.

## 8. Labelling and report (sections 10–11)

- **Labelling.** This follows the Mac-profile spec §6 exactly:
  - units by (item, normalised answer);
  - batches of up to 100 that never split a question; re-export adds only new units;
  - the exact `batch_NN.md` text;
  - human CSV export;
  - import that finds the CSV block, accepts CRLF and quoted notes, and rejects unknown ids, bad labels,
    bad tags, conflicting duplicates and rows with extra fields, storing nothing on any error;
  - source names `claude-manual[-suffix]` or `human:<id>`.

  Batch files go to `notebooks/runs/<profile>/annotation/claude/`. Replies are read from
  `…/annotation/claude/replies/*.txt` (every file found, or a list in the cell).
- **Report.** This follows the Mac-profile spec §7 and its review fixes:
  - accuracy per config, plus unparseable counts and `[k of n judged]`;
  - accuracy per label source;
  - agreement with the first label source: gap in points, κ (None when undefined), FPR and FNR;
  - visual gain per system with a blind partner;
  - the run-2 table on the main set.

  It shows DataFrames inline and writes `runs/<profile>/results.md`, with the note that non-`human:*`
  sources are reference judges.

## 9. Testing

- **Equivalence tests** (`tests/notebook/test_equivalence.py`). They load the notebook with `nbformat`
  and exec the cells tagged `lib` (definitions only) into a namespace. Then they compare against the
  package on the existing fixtures and hand-made inputs:
  - benchmark loading (same items for `tests/fixtures/raw`);
  - frame indices;
  - normalisation;
  - nested sampling and dev/test split (same item sets for the same pool and seed);
  - template filling;
  - verdict parsing;
  - margin readout;
  - unit building and batching;
  - CSV import validation;
  - κ / FPR / FNR.
- **Execution test** (`tests/notebook/test_execute.py`). It builds a tiny synthetic dataset in `tmp_path`:
  fixture annotations plus generated videos. It sets the `parameters` cell to `PROFILE="mac",
  DRY_RUN=True, DATA_ROOT=<tmp>`, executes the notebook with `nbclient` (timeout 120 s), and checks:
  - that every section ran;
  - that `generations.jsonl` and `judgecalls.jsonl` have the expected counts;
  - that batch files exist;
  - that importing a synthetic reply produces `results.md`.
  It also executes with `PROFILE="server", DRY_RUN=True`.
- **Hygiene test.** The notebook contains no `import vjaudit` and no `from vjaudit`, and is saved without
  outputs.
- These tests run in `make sanity` (CPU only). Real models, vLLM and the API are not exercised in tests;
  the first real check is section 8/9 on the Mac.

## 10. Documentation inside the notebook

Every section's markdown has four fixed parts:

- **What** this step does;
- **Why**;
- **Reads / writes**;
- **Mac vs server**: what to change, or "nothing".

Long-running cells say roughly how long they take. Section 0 explains:

- running top to bottom versus jumping to a section;
- that every section first loads what it needs from the results store, so after a restart you can start
  at any section once its inputs exist;
- Run 1 vs Run 2.

## 11. Risks

| Risk | Mitigation |
|---|---|
| The notebook drifts from the tested package | The equivalence tests pin the core functions to the package's behaviour |
| Long runs in a notebook are fragile (kernel restarts, laptop sleep) | Every step is resumable; section 0 recommends `caffeinate` on the Mac and shows how to resume |
| Real-model paths are untested here | The same calls as the reviewed package code; the dry run checks all the plumbing; the first real Mac run starts with a 5-question run (`RUN_LIMIT` in settings) |
| The notebook gets very long | Code cells stay short and each defines only its own step; a collapsible table of contents; outputs are cleared before saving |
