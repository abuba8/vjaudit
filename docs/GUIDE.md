# vjaudit — Codebase Guide and Stage-1 Runbook

This guide explains what the code does, how it is organised, what every file is for, and exactly how to
run Stage 1 on the GPU server. Stage 1 = everything that must exist before OpenAI shuts down
`gpt-3.5-turbo` on **2026-10-23**: the prior-pipeline integrity check, the data samples, every model
answer, and GPT-3.5's verdicts on all of them.

- Spec (why): `docs/superpowers/specs/RESEARCH_DIRECTION.md`, `docs/superpowers/specs/REQUIREMENTS_CONTRACT.md`
- Plan (what was built): `docs/superpowers/plans/2026-10-04-stage1-capture-before-gpt35-shutdown.md`
- Decisions log (dated changes): `docs/decisions.md`

> **On a Mac, or without an OpenAI key?** Read **[docs/EXPLAINED.md](EXPLAINED.md)** first. It explains
> the research idea and the code in plain language and has the complete Mac runbook (small local
> models, local judges, labelling with Claude). This guide covers the GPU server and the file-by-file
> reference.

---

## 1. What the pipeline does

```
 benchmark files ──► loaders ──► availability check ──► nested samples ──► frame cache
 (MSVD-QA, ANet-QA,   Item        (video found +          main 2000 ⊃         16/32 uniform
  NExT-OE)            records      decodable?)            annotated 300 ⊃      frames per clip
                                                          vid 200; dev/test
                                                                 │
                                                                 ▼
                         11 systems × 3 benchmarks ──► vLLM generation ──► Generation records
                         (5 models, 1 verbose,           (greedy, pinned
                          3 blind, 2 frame-ablation)      revisions)
                                                                 │
                                                                 ▼
                         GPT-3.5 judge (T-VCG, T-RUB) ──► logprob margins ──► JudgeCall records
                         stages S1…S7 in priority order    + parsed verdicts     + capture report
                                                                 │
                                                                 ▼
                                                     freeze: sorted exports + API cache tarball
```

Everything is written to an **append-only artifact store** (`artifacts/`), one JSONL record per item,
generation or judge call. Every record has a content-hash key, so every command is **idempotent and
resumable**: re-running a command skips work that is already stored. A crash never corrupts results.

---

## 2. Quick start (any machine, no GPU)

```bash
# 1. install uv (https://docs.astral.sh/uv/) then:
uv python install 3.11
uv sync                      # CPU dependencies only
make sanity                  # runs the CPU test suite (~300 tests, a few seconds)
uv run python -m vjaudit.cli --help
```

**Using conda instead of uv** (Python 3.11–3.13):

```bash
conda activate dl                                  # any environment name
pip install -e ".[mac]" pytest pytest-asyncio      # ".[gpu]" on the GPU server; "." for CPU only
make sanity
python -m vjaudit.cli --help
```

The Makefile decides how to run Python: with an active conda environment other than `base`, or when uv is
not installed, it calls `python`/`pytest` directly; otherwise `uv run`. Force either with `RUN=` or
`RUN="uv run"`. In this guide, drop the `uv run` prefix from commands when you use conda.

`make sanity` excludes tests that need a GPU (`gpu`), the OpenAI API (`api`), the prior pipeline's
cached outputs (`legacy`), Hugging Face Hub access (`hf`) or real models on a Mac (`mac`).

### Profiles: server and mac

All model, judge and sampling settings live in a **profile**, a folder with three files:

| Profile | Folder | For |
|---|---|---|
| `server` (default) | `configs/profiles/server/` | the GPU server: 5 generators on vLLM, GPT-3.5 capture stages S1–S7 |
| `mac` | `configs/profiles/mac/` | a 32 GB Apple-Silicon Mac: 3 small generators and 2 local judges via `transformers` (see `docs/EXPLAINED.md`) |

- Choose one with `make <target> PROFILE=mac`, or `--profile mac` on any CLI command, or the environment
  variable `VJAUDIT_PROFILE`. Every command prints `profile: <name>` first.
- Each profile writes to its own folders: `artifacts/<profile>/`, `artifacts/frames/<profile>/` and
  `artifacts/api_cache/<profile>/`.
- Reports are per profile too, for example `docs/samples_server.md` and `docs/capture_mac.md`.
- **Records written before profiles existed** sit directly in `artifacts/` (`artifacts/item/`,
  `artifacts/generation/`, …) and are not seen by the new layout; commands print a `WARNING` about them.
  Move them into the server profile once, before running anything else (the default paths are shown):

  ```bash
  mkdir -p artifacts/server artifacts/frames/server artifacts/api_cache/server
  for k in item_pool item generation judgecall; do [ -d artifacts/$k ] && mv artifacts/$k artifacts/server/; done
  find artifacts/frames -maxdepth 1 -type f -name '*.npz' -exec mv {} artifacts/frames/server/ \;
  find artifacts/api_cache -mindepth 1 -maxdepth 1 -type d ! -name server ! -name mac \
       -exec mv {} artifacts/api_cache/server/ \;
  ```

---

## 3. Stage-1 runbook (GPU server: 3× A30 24 GB)

Run the steps in order. Each step lists the command, what it does, and what to check. Target dates
come from the plan; **the hard deadline is the GPT-3.5 shutdown on 2026-10-23** (finish capture by Oct 22).

### Step 0 — Install (Oct 4)

```bash
git clone <this repo> vjaudit && cd vjaudit
uv python install 3.11
uv sync --extra gpu          # adds vLLM (Linux only), transformers, huggingface_hub (pinned in uv.lock)
export HF_TOKEN=...          # Hugging Face token (some models/datasets need it)
export OPENAI_API_KEY=...    # needed only for Steps 8–10
make sanity
```

Check: `make sanity` passes. `uv run python -c "import vllm, torch; print(torch.cuda.device_count())"` prints `3`.

### Step 1 — Configure machine-specific paths

```bash
mkdir -p configs/local
cp configs/paths.example.yaml configs/local/paths.yaml     # edit if your data lives elsewhere
```

`configs/local/` is gitignored, so personal paths never end up in the anonymised release.

### Step 2 — Download the datasets (Oct 4–6)

```bash
bash scripts/download/download_all.sh                     # everything (ActivityNet videos ≈130 GB)
# or one at a time:
bash scripts/download/download_msvd.sh                    # needs megatools (apt/brew install megatools)
bash scripts/download/download_nextqa.sh                  # needs the HF CLI (installed with --extra gpu)
bash scripts/download/download_anetqa.sh                  # needs git, HF CLI, unzip, ~260 GB free
# annotations only (fast; for inspecting the files — run Step 4 only after the videos finish):
bash scripts/download/download_all.sh --annotations-only
```

| Benchmark | Annotations (source) | Videos (source) | Lands in |
|---|---|---|---|
| MSVD-QA | `test_qa.json`, `youtube_mapping.txt` (official MEGA links from xudejing/video-question-answering) | `YouTubeClips.tar` from the UT Austin MSVD page (~1.7 GB) | `data/msvd/` |
| ActivityNet-QA | `test_q.json`, `test_a.json` (git clone of MILVLG/activitynet-qa) | `lmms-lab/ActivityNetQA` on Hugging Face (28 zips, ~130 GB) | `data/anetqa/` |
| NExT-OE (+ NExT-QA MC) | `OE/` and `MC/` test parquet (`lmms-lab/NExTQA`) | `videos.zip` from `lmms-lab/NExTQA` (~6.5 GB) | `data/nextqa/` |

All scripts are safe to re-run: finished parts are skipped. `DATA_ROOT=/big/disk bash scripts/download/download_all.sh`
puts data elsewhere; then edit `configs/local/paths.yaml` to match.

### Step 3 — SP0: prior-pipeline integrity check (Oct 4–6, Gate G0)

```bash
cp configs/legacy.example.yaml configs/local/legacy.yaml   # fill in paths to the prior pipeline's outputs
make sp0-report                                            # writes docs/sp0_report.md
uv run pytest tests/legacy -m legacy -v                    # Tables 2–3 must reproduce to ±0.1
```

`docs/sp0_report.md` contains:

1. **Generation identity across conditions.** The percentage of byte-identical generated answers for
   every pair of prompting conditions, on val and test.
2. **Tables 2–3.** The BlackboxNLP tables recomputed from the cached judge outputs, next to the
   published values.
3. **S1 label.** The one-shot exemplar's verdict label (R-SP0-05).

Then do the manual check (R-SP0-02). Read the prior `inference.py` files and decide whether the
exemplars actually reached the **generator**. Write the outcome in `docs/decisions.md` as one of:
(a) cache reuse by design, (b) exemplars never reached the generator, or (c) other.

**If (b): stop and tell all authors.** Their decision on a correction to the BlackboxNLP paper is
Gate G0. The rest of Stage 1 does not depend on it and continues.

### Step 4 — Availability check and samples (Oct 6–9)

```bash
make data-check     # = vjaudit.cli data check-availability → docs/data_availability_server.md
make data-sample    # = vjaudit.cli data sample → docs/samples_server.md
```

`data-check` loads all three benchmarks, finds each item's video (recursive search by file name), and
checks it decodes. It stores the pool as records of kind `item_pool`. Missing or undecodable videos are
**listed, never silently dropped**. If more videos arrive later, re-run with `--force`: the old pool is archived (`artifacts/item_pool.archived-<time>`) and rebuilt. Likewise `data sample --force` re-samples. It prints `WARNING: <benchmark> has N available items < main sample 2000`
if a benchmark cannot fill its sample; that means a video-download problem to fix first.

`data-sample` draws, per benchmark (seed 42):

- the main sample of 2,000 questions;
- the annotated sample of 300 questions, drawn from main;
- the video-labelled subset of 200 questions, drawn from annotated.

Sampling is stratified by question type. MSVD-QA gets ≥ 15 annotated questions per type. Annotated
questions are split 40/60 into dev and test. Check the counts in `docs/samples_server.md`.

### Step 5 — Pin models, freeze golden prompts, smoke test (Oct 8–11)

```bash
uv run python -m vjaudit.cli gen pin-revisions     # writes HF revision SHAs into configs/profiles/server/generators.yaml
uv run python -m vjaudit.cli gen render-golden     # renders the chat-template prompts per model family
#   → inspect tests/generate/golden/*.txt by eye: the video placeholder is present (video mode),
#     absent (blind mode), and the assistant turn is open at the end. Then commit them.
uv run pytest tests/generate -m "hf" -v            # golden prompts match the pinned processors
uv run pytest tests/generate/test_smoke_gpu.py -m gpu -v
git commit -am "chore: pin generator revisions and golden prompts"
```

The smoke test loads each of the 5 generators and answers 5 items with and without video. If a
model runs out of memory at 32 frames on ActivityNet-QA, set its `n_frames: {default: 16, anetqa: 16}`.
If a model fails entirely, substitute a fallback (`lmms-lab/LLaVA-Video-7B-Qwen2` or
`DAMO-NLP-SG/VideoLLaMA3-7B`). **Record either change in `docs/decisions.md`.**
`VllmEngine` refuses to run a model whose revision is still `UNPINNED`.

### Step 6 — Throughput (R-SP0-07)

```bash
make throughput      # = scripts/throughput.py --n 50 → docs/throughput.md (items/s per model, video and blind)
```

### Step 7 — Generate the annotated subset (target Oct 11)

```bash
make generate-annotated        # 3 processes, one per GPU; logs in logs/gen-annotated-{0,1,2}.log
uv run python -m vjaudit.cli gen rerun-check --system qwen3vl8b --n 50   # must print 50/50 identical
```

`generate-annotated` ends by printing coverage. Every line must be `missing 0`, and the run produces
about 9,900 generations: 900 questions × 11 systems. The 11 systems are:

| System id | Meaning |
|---|---|
| `qwen25vl7b`, `qwen3vl8b`, `qwen3vl2b`, `internvl35_8b`, `llavanextvideo7b` | short answers, 16 frames (32 on ActivityNet-QA) |
| `qwen3vl8b-sentence` | same model, one-sentence answers with justification (verbosity probe) |
| `qwen25vl7b-blind`, `qwen3vl8b-blind`, `internvl35_8b-blind` | question only, no video |
| `qwen3vl8b-f1`, `qwen3vl8b-f4` | frame-count ablation (1 and 4 frames) |

If `rerun-check` reports mismatches, record the vLLM version in `docs/decisions.md` and investigate
before continuing. Prefix caching for generation is already off.

### Step 8 — Verify the judge prompts (before any API call)

```bash
bash scripts/download/fetch_upstream_prompts.sh
# → clones Video-ChatGPT and runs: vjaudit.cli judge verify-vcg --upstream <scripts...> --commit <sha>
uv run python -m vjaudit.cli judge vendor-rub \
    --prior <prior repo>/src/Prompts/LLM_as_judge_base.j2 \
    --var-map <prior question var>=question,<prior ground-truth var>=answer,<prior generated var>=pred \
    --commit <prior repo sha>
git commit -am "chore: verify T-VCG and vendor T-RUB"
```

- **T-VCG** is the official Video-ChatGPT GPT-3.5 prompt (yes/no + 0–5 score). `verify-vcg` reads the
  upstream script's `messages=[...]` with Python's AST and requires an exact match with
  `configs/prompts/judge/t_vcg_{system,user}.txt`. It then marks `t-vcg` verified in
  `configs/prompts/judge/source.yaml`. If the MSVD/MSRVTT/TGIF scripts differ, they are printed;
  note the difference in `docs/decisions.md`.
- **T-RUB** is the BlackboxNLP three-class rubric. The shipped `t_rub.txt` is a reconstruction from the
  paper. `vendor-rub` replaces it with the prior repo's real template in zero-shot form. Open the
  `.j2` file to find its variable names for `--var-map`.
- `judge capture` refuses an unverified template unless you pass `--allow-unverified`.

### Step 9 — GPT-3.5 capture, annotated subset first (target S1–S2 by Oct 13)

```bash
make capture-gpt35 STAGES=S1,S2                 # dry run: prints jobs and a cost estimate per stage
make capture-gpt35 STAGES=S1,S2 CONFIRM=1       # sends requests; writes docs/capture_server.md
```

Check `docs/capture_server.md` after S1:

- the **ok rate** must be ≥ 0.98;
- the **api_no_logprob rate** must be ≤ 0.05.

If either fails, stop. Fix the parser or readout; every response is cached, so re-parsing costs
nothing and needs no new API calls. Then continue.

The capture stages (pinned in `configs/profiles/server/grid.yaml`, run strictly in this order):

| Stage | Items | Systems | Template | Decoding |
|---|---|---|---|---|
| S1 | annotated | 9 core | T-VCG | temperature 0, logprobs top-20 |
| S2 | annotated | 9 core | T-RUB | temperature 0, 1 token, logprobs top-20 |
| S3 | main | 9 core | T-VCG | as S1 |
| S4 | main | 9 core | T-RUB | as S2 |
| S5 | annotated | 9 core | T-VCG | API default temperature (as in the official script), 3 samples |
| S6a–d | annotated, then main | 2 ablation | T-VCG, T-RUB | as S1/S2 |
| S7 | annotated | 9 core | T-VCG | as S1, judge `gpt-3.5-turbo-1106` (snapshot variance) |

Spend is capped per judge by `max_usd` in `configs/profiles/server/judges.yaml` (100 USD and 20 USD). If a request
returns `model_not_found`, capture stops cleanly, keeps everything captured, and appends an entry to
`docs/decisions.md`.

### Step 10 — Main sample (target generation Oct 15, capture S3–S7 by Oct 21)

```bash
make generate-main                               # ≈56,100 more generations
make capture-gpt35 STAGES=S3,S4 CONFIRM=1
make capture-gpt35 STAGES=S5,S6a,S6b,S6c,S6d,S7 CONFIRM=1
```

**Fallback gate:** if main-sample generation is not done by Oct 15, run S5 on the annotated subset
now. Then run S3–S4 on whatever main-sample generations exist; jobs run in `gen_id` order and
everything is resumable.

### Step 11 — Judge-free anchor (any time after Step 10)

```bash
make generate-mc       # NExT multiple-choice answers for the 5 core models → docs/mc_anchor.md
```

### Step 12 — Freeze (by Oct 22)

```bash
make freeze            # artifacts/freeze-stage1/{item,generation,judgecall}.jsonl + api_cache.tar.gz
git add artifacts/manifest.jsonl && git commit -m "chore: freeze stage-1 artifacts"
```

Copy `artifacts/freeze-stage1/` to **two** storage locations. Record the paths and the sha256 values
from `artifacts/manifest.jsonl` in `docs/stage1_freeze.md`. Add a `docs/decisions.md` entry with the
final coverage per capture stage.

---

## 4. How the code is organised

### 4.1 Layers

| Layer | Package | Responsibility |
|---|---|---|
| Foundations | `hashing`, `config`, `paths`, `profile`, `device`, `schemas`, `store` | ids and hashes, YAML config models, machine paths, profiles, compute device, record types, the artifact store |
| Data | `vjaudit.data` | load benchmarks → `Item`; find and check videos; stratified samples; frame cache |
| Generation | `vjaudit.generate` | systems, prompt adapters per model family, vLLM engine (server), transformers engine (Mac), runner, coverage, smoke check |
| Judging | `vjaudit.judge` | judge templates and provenance, output parsers, logprob readout, OpenAI client, local judges (Mac), capture, smoke check |
| Labelling | `vjaudit.annotate` | answer units, batch export for Claude or people, validated label import |
| Reporting | `vjaudit.report` | κ, error rates, judge-vs-reference tables, visual gain |
| Integrity | `vjaudit.legacy` | SP0 checks on the prior (BlackboxNLP) pipeline's outputs |
| Release | `freeze` | deterministic exports and the manifest |
| Interface | `cli*.py`, `Makefile`, `scripts/` | commands people run |

### 4.2 Record types (in `src/vjaudit/schemas.py`)

Every record also carries `config_hash`, `git_sha` and `created_at`.

| Kind (store directory) | Key | What it is |
|---|---|---|
| `item_pool` | `item_id` | every benchmark question after the availability check |
| `item` | `item_id` | the same, with sample flags (`in_main`, `in_annotated`, `in_vid_subset`, `vid_rank`, `human_split`) |
| `generation` | `gen_id` | one model answer for one (item, system), with frame indices and decoding settings |
| `judgecall` | `judge_call_id` | one judge verdict: parsed verdict, VCG score, logprob margin, top tokens, status, cost |

IDs:

- `item_id = "<benchmark>:<split>:<native id>"`
- `gen_id = sha1(item_id | system_id | config_hash(generator + system + prompt + decoding))[:16]`
- `judge_call_id = sha1(gen_id | judge | protocol_id | "none" | "text" | sample_idx)[:16]`
- `protocol_id = "<template>@<template-hash[:8]>/<decoding tag>"`

Changing a prompt, a model revision or decoding therefore creates **new** ids; old results are never
overwritten.

### 4.3 The artifact store (`src/vjaudit/store.py`)

`artifacts/<kind>/part-<pid>-<n>.jsonl`, one line per record, flushed and fsynced. Each process writes
its own part file, so the 3 GPU processes never collide. On start-up, the store loads all keys.
`put()` returns `False` and writes nothing if the key exists, which is what makes re-runs skip finished
work. A truncated last line, left by a crash, is ignored with a warning. `export_sorted()` gives a
canonical, order-independent file (sorted by key, without `created_at`), used to compare runs and to
freeze.

### 4.4 Judge readout

- **T-VCG** asks GPT-3.5 for a dict string like `{'pred': 'yes', 'score': 4}`.
  - `parse_vcg` extracts the verdict and score, with a regex fallback.
  - `readout_vcg` finds the token that carries the `pred` value and computes
    `margin = logsumexp(yes-like tokens) − logsumexp(no-like tokens)` from the top-20 logprobs at that position.
- **T-RUB** answers with a single Yes/No token, so its margin is read at the first token.
- **`parse_status` precedence:** `unparseable` > `off_vocab` > `api_no_logprob` > `ok`. The parsed
  verdict is always stored, even when no margin is available.

---

## 5. File-by-file reference

### Top level

| File | Purpose |
|---|---|
| `pyproject.toml` | package metadata; CPU dependencies; `gpu` extra (vLLM on Linux, transformers, huggingface_hub); `mac` extra (torch, torchvision, transformers, accelerate, huggingface_hub); pytest markers |
| `uv.lock` | exact versions of every dependency, including both extras |
| `Makefile` | one-word commands for every runbook step; `PROFILE=mac` selects the Mac profile. Server: `sanity`, `data-check`, `data-sample`, `generate-*`, `capture-gpt35`, `freeze`, `sp0-report`, `throughput`. Mac: `pin`, `smoke`, `run1`, `run2`, `judge-run1`, `judge-run2`, `annotate-export`, `annotate-import`, `annotate-status`, `report` |
| `.github/workflows/ci.yml` | CI: CPU test suite on every push |
| `.gitignore` | ignores `/data/`, `artifacts/*` (except the manifest), `configs/local/*.yaml`, `/annotation/`, logs |

### `configs/`

| File | Purpose |
|---|---|
| `benchmarks.yaml` | per benchmark: split, raw file names, which `paths.yaml` keys hold its annotations and videos |
| `profiles/server/generators.yaml` | the 5 server generators: HF repo, pinned revision, model family, frames per benchmark, styles, blind / frame-ablation variants |
| `profiles/server/judges.yaml` | API judges (`gpt-3.5-turbo-0125`, `-1106`): price per 1M tokens, spend cap, concurrency |
| `profiles/server/grid.yaml` | seed, sample sizes, MSVD per-type minimum, dev fraction, and the 10 capture stages in order |
| `profiles/mac/generators.yaml` | the 3 small Mac generators (`backend: hf`, 8 frames, `dtype: auto`) |
| `profiles/mac/judges.yaml` | local judges Qwen3-4B and Qwen2.5-1.5B (`backend: transformers`, pinned revision), plus GPT-3.5 for the optional stages |
| `profiles/mac/grid.yaml` | Mac samples (1,000 / 100 / 30), `frame_max_side: 448`, stages M1–M6 and optional G1–G2 |
| `paths.example.yaml` | template for `configs/local/paths.yaml` (data, artifacts, frame cache, API cache locations) |
| `legacy.example.yaml` | template for `configs/local/legacy.yaml` (where the prior pipeline's outputs are) |
| `prompts/generation/{short,sentence,mc}.txt` | the exact generation prompts (`{question}` and `{a0}`…`{a4}` slots) |
| `prompts/judge/t_vcg_system.txt`, `t_vcg_user.txt` | the Video-ChatGPT judge prompt, verbatim |
| `prompts/judge/t_rub.txt` | the BlackboxNLP rubric judge prompt (zero-shot) |
| `prompts/judge/source.yaml`, `SOURCE.md` | where each judge prompt came from and whether it has been verified |

### `src/vjaudit/` — foundations

| File | Purpose |
|---|---|
| `hashing.py` | `canonical_json`, `config_hash` (sha256), `short_id` (sha1[:16]), `git_sha` |
| `config.py` | pydantic models for every YAML file; `load_yaml(path, Model)` |
| `paths.py` | `load_paths()` for `configs/local/paths.yaml`; `REPO_ROOT`, `CONFIGS` |
| `profile.py` | `load_profile()` → `Profile` (generators, judges, grid, per-profile artifact/cache folders); `VJAUDIT_PROFILE` |
| `device.py` | `pick_device()` (mps → cuda → cpu) and `default_dtype()` |
| `schemas.py` | `Item`, `Generation`, `JudgeCall`, `Unit`, `Label` record models (contract §8.2 + Stage-1 additions) |
| `store.py` | `ArtifactStore` (append-only, resumable, content-addressed) and `stamp()` provenance |
| `normalize.py` | `normalize_answer()` for de-duplicating answers into annotation units |
| `freeze.py` | `freeze()`: sorted exports, deterministic API-cache tarball, manifest |
| `cli.py` | root command `python -m vjaudit.cli` with groups `data`, `gen`, `judge`, `annotate`, `report` |
| `cli_common.py` | the shared `--profile` option; prints `profile: <name>`; per-profile report paths |
| `cli_data.py` | `data check-availability`, `data sample` |
| `cli_gen.py` | `gen pin-revisions`, `render-golden`, `smoke`, `run`, `coverage`, `rerun-check`, `mc-report` |
| `cli_judge.py` | `judge verify-vcg`, `vendor-rub`, `smoke`, `capture` |
| `cli_annotate.py` | `annotate export`, `import`, `status` |
| `cli_report.py` | `report summary` |

### `src/vjaudit/data/`

| File | Purpose |
|---|---|
| `loaders/msvd.py` | MSVD-QA `test_qa.json` + `youtube_mapping.txt` → items; question type = first word (what/who/how/when/where/other) |
| `loaders/anetqa.py` | ActivityNet-QA `test_q.json` + `test_a.json` joined on `question_id`; type codes 0–8 → names |
| `loaders/nextqa.py` | NExT-OE test parquet, joined with NExT-QA MC on (video, qid) to attach the 5 options |
| `loaders/_common.py` | builds stamped `Item`s; rejects duplicate question ids |
| `availability.py` | finds each item's video file by name (recursive), probes that it decodes, writes the availability report |
| `sampling.py` | largest-remainder stratified allocation; nested main ⊃ annotated ⊃ vid samples; `vid_rank`; dev/test split |
| `frames.py` | `sample_indices` (uniform, segment-centred); `FrameCache` (decode once, store `.npz`, max side 768 px) |

### `src/vjaudit/generate/`

| File | Purpose |
|---|---|
| `systems.py` | `SystemSpec`; `expand_systems` (the 11 judged systems); `expand_mc_systems`; `parse_mc_letter` |
| `adapters.py` | per-family message formats (Qwen-VL, InternVL, LLaVA-NeXT-Video, LLaVA-OneVision); prompt filling; `HFChatRenderer` (pinned chat template); `golden_render` |
| `engine.py` | `VllmEngine` (greedy, pinned revision, 1 video per prompt) and `FakeEngine` (tests) |
| `hf_engine.py` | `TransformersEngine`: greedy generation with `transformers` on MPS/CUDA/CPU, one request at a time; `close()` frees memory |
| `factory.py` | `make_engine` (backend `vllm` or `hf`) and `make_renderer` |
| `smoke.py` | `smoke_generators`: a few answers per generator, video and blind, with timings |
| `runner.py` | `run_generation`: one `Generation` per (item, system), resumable, batched |
| `coverage.py` | `run_subset` (sharded across GPUs; one engine at a time, closed before the next), `coverage`, `mc_accuracy`, `rerun_check` |
| `pin.py` | `pin_revisions`: writes HF revision SHAs into a profile's `generators.yaml` / `judges.yaml`, keeping comments |

### `src/vjaudit/judge/`

| File | Purpose |
|---|---|
| `templates.py` | loads T-VCG / T-RUB; one-pass slot filling; `protocol_id`; `verify_vcg_source` (AST check against upstream); `vendor_rub` |
| `parse.py` | `parse_vcg` (dict string → verdict, score, status); `parse_yes_no` |
| `readout.py` | `token_class`, `margin_from_top`, `vcg_verdict_index`, `readout_vcg`, `readout_first_token`, `p_yes` |
| `api_openai.py` | `OpenAIJudgeClient`: request-hash cache on disk, retries with backoff, spend cap, `ModelRetired` on `model_not_found` |
| `local.py` | `content_from_logits`, `HFCausalBackend`, `LocalTransport`: a local judge that returns OpenAI-shaped replies with top-20 logprobs |
| `factory.py` | `make_client`: OpenAI transport or `LocalTransport` (cache per judge and revision) |
| `capture.py` | `CaptureStage` (incl. `optional`), `select_stages`, `build_jobs` (priority order, skip done), `run_capture` (stops cleanly), `write_capture_report` |
| `smoke.py` | `smoke_judges`: one T-VCG and one T-RUB call per local judge |

### `src/vjaudit/annotate/`

| File | Purpose |
|---|---|
| `units.py` | `build_units` (one unit per question and normalised answer), `plan_batches` (never splits a question) |
| `export.py` | `export_claude` (`batch_NN.md` to paste into Claude), `export_human` (`batch_NN.csv`), `manifest.json`; re-export adds only new units |
| `importer.py` | `parse_label_csv` (finds the CSV block in a reply), `import_labels` (all-or-nothing validation), `validate_source` |

### `src/vjaudit/report/`

| File | Purpose |
|---|---|
| `metrics.py` | `cohen_kappa`, `error_rates` (FPR/FNR), `yes_rate` |
| `summary.py` | `build_tables`, `visual_gain`, `write_summary` → `docs/results_<profile>.md` |

### `src/vjaudit/legacy/` (SP0)

| File | Purpose |
|---|---|
| `load.py` | reads `configs/local/legacy.yaml`; `LegacyCell`; finds the S1 one-shot verdict label |
| `diff.py` | `pairwise_identity`: % byte-identical generations between prompting conditions |
| `tables.py` | `acceptance`, `table3` (test acceptance), `table2` (judge vs. human consensus: J%, FP, FN) |
| `report.py` | writes `docs/sp0_report.md` |

### `scripts/`

| File | Purpose |
|---|---|
| `download/common.sh` | shared helpers: HF and MEGA download, unzip, logging |
| `download/download_msvd.sh` | MSVD-QA annotations (MEGA) + YouTubeClips.tar |
| `download/download_anetqa.sh` | ActivityNet-QA annotations (GitHub) + videos (lmms-lab, ~130 GB) |
| `download/download_nextqa.sh` | NExT-OE/MC annotations + videos (lmms-lab) |
| `download/download_all.sh` | all three, plus creates `configs/local/paths.yaml` |
| `download/fetch_upstream_prompts.sh` | clones Video-ChatGPT and verifies the T-VCG prompt |
| `sp0_report.py` | SP0 report (wrapper around `vjaudit.legacy.report`) |
| `throughput.py` | items/s per generator, video vs blind → `docs/throughput.md` |
| `freeze_stage1.py` | freeze wrapper → `artifacts/freeze-stage1/` + `artifacts/manifest.jsonl` |

### `tests/`

| Path | What it covers |
|---|---|
| `test_hashing.py`, `test_config.py`, `test_schemas.py`, `test_store.py`, `test_normalize.py`, `test_freeze.py` | foundations (incl. crash/resume equivalence, gitignore regression) |
| `data/` | loaders on fixture files, availability (missing vs undecodable), sampling (sizes, nesting, MSVD minimum, 40/60, determinism), frame cache (exact frames, cache hit, corrupt file), data CLI |
| `generate/` | systems, adapters (video/blind, literal braces), runner (resume, empty outputs, blind), coverage/sharding/MC/rerun-check, CLI, pinning; `golden` (`hf`) and GPU smoke (`gpu`) |
| `judge/` | templates (frozen hash, upstream AST check, vendoring), parsers, readout (split tokens), API client (cache, retry, retirement, budget, concurrency), capture (order, resume, S5 samples, statuses, retirement, S7), capture CLI |
| `legacy/` | SP0 tools on fixtures; `legacy`-marked regression to the published tables |
| `scripts/` | download scripts parse, `--help` works offline, official URLs present |
| `annotate/`, `report/` | units, export privacy, import validation (fences, CRLF, quoted notes, invented ids), CLI round trip; κ, FPR/FNR, gap, visual gain |
| `test_profile.py`, `test_tooling.py`, `test_docs.py` | profiles and per-profile folders; Makefile/pyproject wiring; docs name only real targets, commands and paths |
| `test_mac_real.py` | real Mac models (`mac` marker; `uv run pytest -m mac` on the Mac) |
| `test_e2e.py` | the whole chain on tiny synthetic videos with a fake model and fake API |
| `fixtures/` | small raw benchmark files, legacy-format JSONs, upstream-format script, OpenAI-format logprob payloads |
| `helpers/factories.py` | valid `Item` / `Generation` / `JudgeCall` builders for tests |

---

## 6. Running the tests

```bash
make sanity                                  # CPU suite (what CI runs)
uv run pytest tests/legacy -m legacy         # needs configs/local/legacy.yaml
uv run pytest -m hf                          # needs HF Hub + rendered golden prompts
uv run pytest -m gpu                         # needs CUDA + the gpu extra
uv run pytest -m mac                         # on the Mac: loads Qwen3-VL-2B and Qwen2.5-1.5B (mac extra)
uv run pytest -m api -k live_smoke           # one real GPT-3.5 call (needs OPENAI_API_KEY)
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `configs/local/paths.yaml not found` | Step 1 |
| `'item_pool' already exists … re-run with --force` | intended: derived pools are rebuilt, not mixed; add `--force` |
| `FAILED <stage> <gen_id>: 400 …` after capture | that request failed for good (e.g. content filter); the rest continued; re-run to retry |
| `… skipped: <video_id>: cannot decode` in generation logs | that video fails a full decode; the item stays missing in `gen coverage`; fix or replace the file and re-run |
| `WARNING: anetqa has N available items < main sample 2000` | videos missing or not extracted; see `docs/data_availability_server.md` |
| `VllmEngine … revision is UNPINNED` | run `gen pin-revisions` (Step 5) |
| `t-vcg is not verified` | Step 8 |
| CUDA out of memory at 32 frames | set that model's `anetqa` frames to 16 and record it in `docs/decisions.md` |
| `rerun-check` mismatches | record the vLLM version; make sure nothing else runs on that GPU; re-check |
| capture stops with `BudgetExceeded` | raise `max_usd` in `configs/profiles/server/judges.yaml` deliberately, then re-run (finished calls are skipped) |
| capture stops with `ModelRetired` | the snapshot is gone; a decisions entry was written; freeze what exists (Step 12) |
| megatools missing | `sudo apt install megatools` / `brew install megatools` |
| `hf: command not found` | `uv sync --extra gpu` (provides `hf`) or `uv tool install huggingface_hub` |

---

## 8. What was verified where

- **Verified in this repository (CPU):** every module above, through its tests and the end-to-end
  test on synthetic videos.
- **Must be done on the GPU server, because it needs GPUs, the HF Hub, the OpenAI API, the real
  videos or the legacy outputs:**
  - Step 3 (SP0 run and decision);
  - the real benchmark files (Step 4 checks the loaders against them);
  - revision pinning and golden prompts (Step 5);
  - the GPU smoke test and throughput;
  - generation runs;
  - prompt verification against upstream (Step 8);
  - the live API smoke test and the capture;
  - freezing.

The download scripts point at sources checked on 2026-10-04 but were not run end-to-end from here.
