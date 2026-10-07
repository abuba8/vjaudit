# Differences between v1 and v2

| | v1 | v2 |
|---|---|---|
| Folder | `Topic C/vjaudit` | `Topic C/vjauditv2` |
| Git branch | `stage1` | `mac-profile` |
| Commit | `f5548e0` (plus your local, uncommitted fix to `scripts/download/download_nextqa.sh`) | `mac-profile`: all commits since `f5548e0` (list them with `git log f5548e0..HEAD`) |
| Runs on | the GPU server (3× A30, vLLM) | the GPU server **and** a 32 GB Apple-Silicon Mac |
| Judges | GPT-3.5 through the OpenAI API (needs a key) | the same, **or** small local judges (no key, no cost) |
| Install | uv, Python 3.11 | conda (e.g. `dl`) or uv, Python 3.11–3.13 |
| CPU tests | 202 | 318 |

**Size of the change:** 85 files: 45 added, 37 modified, 3 moved;
6,004 lines added and 444 removed. This file (`differences.md`) is not counted.

**Data:** v2 does not copy the datasets. Its `configs/local/paths.yaml` points to `../vjaudit/data/`.

**See every changed line yourself:** v2 contains v1's history, so inside `vjauditv2` run
`git diff f5548e0 HEAD` (all changes) or `git diff f5548e0 HEAD -- <file>` (one file).

---

## 1. What changes for you, area by area

| Area | v1 | v2 |
|---|---|---|
| **Where settings live** | `configs/generators.yaml`, `configs/judges.yaml`, `configs/grids/stage1.yaml` | `configs/profiles/server/{generators,judges,grid}.yaml` (same content, moved) and new `configs/profiles/mac/` |
| **Choosing a set-up** | only one (the server) | `make <target> PROFILE=mac`, `--profile mac`, or the variable `VJAUDIT_PROFILE`; the default is `server`. Every command prints `profile: <name>` first |
| **Where results are stored** | `artifacts/`, `artifacts/frames/`, `artifacts/api_cache/` | one folder per profile: `artifacts/<profile>/`, `artifacts/frames/<profile>/`, `artifacts/api_cache/<profile>/`. Commands warn if records are still in the v1 layout |
| **Report file names** | `docs/data_availability.md`, `docs/samples.md`, `docs/gpt35_capture.md` | `docs/data_availability_<profile>.md`, `docs/samples_<profile>.md`, `docs/capture_<profile>.md`, and new `docs/results_<profile>.md` |
| **Command options removed** | `data check-availability --grid`, `data sample --grid`, `judge capture --grid --judges` | replaced by `--profile` |
| **New commands** | — | `gen smoke`, `judge smoke`, `annotate export`, `annotate import`, `annotate status`, `report summary` |
| **Changed commands** | — | `gen run`: picks vLLM or transformers from the config, `--allow-unpinned`. `gen pin-revisions`: pins the profile's models **and** local judges. `judge capture`: local judges, optional stages run only when named, unknown stage names are rejected, `--allow-unpinned` |
| **How answers are generated** | vLLM only | vLLM (`backend: vllm`) or Hugging Face transformers on the Apple GPU / CUDA / CPU (`backend: hf`); one model in memory at a time; pure greedy decoding |
| **Model families** | Qwen-VL, InternVL, LLaVA-NeXT-Video | the same plus LLaVA-OneVision |
| **Generation IDs** | frame size not included | frame size included when it is not 768 px (server IDs are unchanged) |
| **Judges** | OpenAI API only | also local models (`backend: transformers`) that return the same format as the API, including top-20 token probabilities; cached per judge and revision; one local judge runs at a time |
| **Judge records** | `judge_rev` = model name | `judge_rev` = pinned revision for local judges (model name for API judges, as before) |
| **Judge stages** | S1–S7 (GPT-3.5) | server: S1–S7 unchanged. Mac: M1–M6 (local judges) and optional G1–G2 (GPT-3.5) |
| **Labelling** | not in the code | new: answer units, ready-to-paste batches for Claude, CSV sheets for people, validated all-or-nothing import, `Unit` and `Label` records. Claude's labels are a reference judge, never called human |
| **Results report** | capture report and MC anchor only | new: accuracy per system under each judge and label source, gap, Cohen's κ, FPR/FNR, visual gain |
| **Makefile** | always `uv run`; server targets only | uses the active conda environment (any except `base`) or uv; `PROFILE`; new Mac targets; `ALLOW_UNPINNED=1`, `ALLOW_UNVERIFIED=1` |
| **Dependencies** | Python 3.11; extra `gpu` | Python 3.11–3.13; `gpu` (vLLM only on Linux) and new `mac` extra |
| **Download scripts** | call `uv run` | use the active conda environment, otherwise uv; NExT-QA script fix included |
| **Docs** | `README.md`, `docs/GUIDE.md` | both updated; new `docs/EXPLAINED.md` (the plain-language guide) and `WORKFLOW.md` (workflow through the code, and the commands) |

## 2. What did not change

- The datasets and their loaders, the sampling algorithm, and frame extraction.
- The generation prompts and the judge prompts (T-VCG, T-RUB).
- Answer normalisation, verdict parsing and the yes/no margin readout.
- The record store format, and the IDs of server generations and judge calls.
- The server's five generators and its GPT-3.5 stages S1–S7.
- The SP0 integrity-check tools (`src/vjaudit/legacy/`) and the freeze logic.

---

## 3. File by file

### 3.1 Added (45)

| File | What it is | Lines |
|---|---|---|
| `WORKFLOW.md` | the workflow traced step by step through the code, and the exact commands to run it | +376 / −0 |
| `configs/profiles/mac/generators.yaml` | Mac video models: Qwen2.5-VL-3B (short, sentence, blind), Qwen3-VL-2B, LLaVA-OneVision-0.5B; `backend: hf`, 8 frames, `dtype: auto` | +38 / −0 |
| `configs/profiles/mac/grid.yaml` | Mac samples (main 1,000 / annotated 100 / vid 30 per benchmark), `frame_max_side: 448`, judge stages M1–M6 and optional G1–G2 | +27 / −0 |
| `configs/profiles/mac/judges.yaml` | local judges Qwen3-4B-Instruct-2507 and Qwen2.5-1.5B-Instruct (`backend: transformers`), plus GPT-3.5 for the optional stages | +29 / −0 |
| `docs/EXPLAINED.md` | plain-language explanation of the idea and the code, and the step-by-step Mac runbook | +909 / −0 |
| `docs/superpowers/plans/2026-10-06-mac-profile-execution-ledger.md` | every decision made while building v2, review fixes and deferred items | +85 / −0 |
| `docs/superpowers/plans/2026-10-06-mac-profile.md` | implementation plan for v2 | +495 / −0 |
| `docs/superpowers/specs/2026-10-06-mac-profile-design.md` | design spec for v2 | +218 / −0 |
| `src/vjaudit/annotate/__init__.py` | new package: labelling | +1 / −0 |
| `src/vjaudit/annotate/export.py` | writes `batch_NN.md` files for Claude and `batch_NN.csv` sheets for people, plus `manifest.json` | +128 / −0 |
| `src/vjaudit/annotate/importer.py` | reads Claude's reply or a filled sheet, validates every row, stores labels only if all rows are valid | +115 / −0 |
| `src/vjaudit/annotate/units.py` | groups answers into units (one per question and normalised answer); plans batches without splitting a question | +44 / −0 |
| `src/vjaudit/cli_annotate.py` | commands `annotate export`, `annotate import`, `annotate status` | +117 / −0 |
| `src/vjaudit/cli_common.py` | the shared `--profile` option; prints `profile: <name>`; warns about records in the old folder layout | +28 / −0 |
| `src/vjaudit/cli_report.py` | command `report summary` | +26 / −0 |
| `src/vjaudit/device.py` | picks the device (Apple GPU `mps`, then `cuda`, then `cpu`) and its default precision | +22 / −0 |
| `src/vjaudit/generate/factory.py` | chooses the generation engine from `backend` (`vllm` or `hf`) | +22 / −0 |
| `src/vjaudit/generate/hf_engine.py` | `TransformersEngine`: runs a video model with Hugging Face transformers on the Mac GPU | +87 / −0 |
| `src/vjaudit/generate/smoke.py` | the quick check behind `gen smoke` | +41 / −0 |
| `src/vjaudit/judge/factory.py` | builds the judge client: OpenAI API or a local model | +18 / −0 |
| `src/vjaudit/judge/local.py` | `LocalTransport`: a local judge model that answers in the OpenAI format, with top-20 token probabilities | +148 / −0 |
| `src/vjaudit/judge/smoke.py` | the quick check behind `judge smoke` | +45 / −0 |
| `src/vjaudit/profile.py` | loads a profile (`configs/profiles/<name>/`) and its per-profile result folders | +78 / −0 |
| `src/vjaudit/report/__init__.py` | new package: reporting | +1 / −0 |
| `src/vjaudit/report/metrics.py` | Cohen's κ, false-positive and false-negative rates, yes-rate | +26 / −0 |
| `src/vjaudit/report/summary.py` | result tables per system, judge vs label source, visual gain → `docs/results_<profile>.md` | +195 / −0 |
| `tests/annotate/__init__.py` | tests for the labelling package | +0 / −0 |
| `tests/annotate/helpers.py` | tests for the labelling package | +10 / −0 |
| `tests/annotate/test_cli_annotate.py` | tests for the labelling package | +58 / −0 |
| `tests/annotate/test_export.py` | tests for the labelling package | +52 / −0 |
| `tests/annotate/test_import.py` | tests for the labelling package | +86 / −0 |
| `tests/annotate/test_units.py` | tests for the labelling package | +42 / −0 |
| `tests/conftest.py` | shared test fixtures (temporary paths file; ignores a `VJAUDIT_PROFILE` set in your shell) | +24 / −0 |
| `tests/generate/test_factory.py` | tests for engine choice, one-model-at-a-time, LLaVA-OneVision adapter, `gen smoke` | +87 / −0 |
| `tests/generate/test_hf_engine.py` | tests for the Mac engine (with stand-in models) | +136 / −0 |
| `tests/judge/test_capture_local.py` | tests for judging with local judges and optional stages | +92 / −0 |
| `tests/judge/test_local.py` | tests for the local judge (token probabilities, margins, errors, caching) | +199 / −0 |
| `tests/report/__init__.py` | tests for the report package | +0 / −0 |
| `tests/report/test_cli_report.py` | tests for the report package | +14 / −0 |
| `tests/report/test_metrics.py` | tests for the report package | +25 / −0 |
| `tests/report/test_summary.py` | tests for the report package | +116 / −0 |
| `tests/test_docs.py` | checks that README, GUIDE, EXPLAINED and WORKFLOW name only real make targets, commands and file paths | +37 / −0 |
| `tests/test_mac_real.py` | real-model checks for the Mac (marker `mac`; not part of `make sanity`) | +39 / −0 |
| `tests/test_profile.py` | profiles and per-profile folders | +86 / −0 |
| `tests/test_tooling.py` | Makefile and pyproject wiring (profiles, conda/uv rule, Python versions) | +89 / −0 |

### 3.2 Moved (3, content unchanged)

| v1 path | v2 path |
|---|---|
| `configs/generators.yaml` | `configs/profiles/server/generators.yaml` |
| `configs/grids/stage1.yaml` | `configs/profiles/server/grid.yaml` |
| `configs/judges.yaml` | `configs/profiles/server/judges.yaml` |

### 3.3 Modified (37)

| File | What changed | Lines |
|---|---|---|
| `.gitignore` | adds `/annotation/` (exported label batches hold dataset text) | +1 / −0 |
| `Makefile` | profile selection (`PROFILE=mac`); runs Python from the active conda environment or uv (`RUN`); new targets `pin`, `smoke`, `run1`, `run2`, `judge-run1`, `judge-run2`, `annotate-export`, `annotate-import`, `annotate-status`, `report`; `ALLOW_UNPINNED` / `ALLOW_UNVERIFIED` flags; `make sanity` also skips `mac` tests | +73 / −14 |
| `README.md` | install with conda (`dl`) or uv; first commands on a Mac; links to `docs/EXPLAINED.md` and `WORKFLOW.md` | +41 / −6 |
| `docs/GUIDE.md` | profiles section, conda install, moved config paths, per-profile report names, new files, moving old results | +110 / −31 |
| `docs/decisions.md` | dated entry for the Mac set-up | +10 / −0 |
| `pyproject.toml` | Python 3.11–3.13 (was 3.11 only); new `mac` extra (torch, torchvision, transformers>=4.57, accelerate, huggingface_hub); vLLM only on Linux; pytest marker `mac` | +12 / −3 |
| `scripts/download/common.sh` | new `run_python` helper: uses the active conda environment, otherwise uv | +10 / −0 |
| `scripts/download/download_all.sh` | final hint now says `make data-check` | +1 / −1 |
| `scripts/download/download_nextqa.sh` | repeated `--include` per pattern (the same fix you had made locally in v1, uncommitted) | +1 / −1 |
| `scripts/download/fetch_upstream_prompts.sh` | runs Python through `run_python` (works in conda) | +3 / −2 |
| `scripts/freeze_stage1.py` | reads the active profile; non-server profiles write `artifacts/manifest_<profile>.jsonl` | +5 / −5 |
| `scripts/throughput.py` | reads the active profile instead of `configs/generators.yaml` | +5 / −7 |
| `src/vjaudit/cli.py` | registers the new `annotate` and `report` command groups | +4 / −0 |
| `src/vjaudit/cli_data.py` | `--profile` replaces `--grid`; per-profile store and report names | +15 / −9 |
| `src/vjaudit/cli_gen.py` | `--profile` on every command; new `gen smoke`; `gen run` picks the engine from the config and accepts `--allow-unpinned`; `pin-revisions` pins the profile's models and judges | +78 / −36 |
| `src/vjaudit/cli_judge.py` | `--profile` replaces `--grid` and `--judges`; new `judge smoke`; capture uses local or API judges, skips optional stages unless named, rejects unknown stage names, `--allow-unpinned` | +56 / −29 |
| `src/vjaudit/config.py` | new family `llava-onevision`; `GridCfg.frame_max_side`; `JudgeCfg.backend`, `revision`, `chat_template_kwargs` | +5 / −1 |
| `src/vjaudit/generate/adapters.py` | `LlavaOneVisionAdapter`; golden prompts read the server profile | +10 / −2 |
| `src/vjaudit/generate/coverage.py` | closes each model before loading the next; the renderer is built from the loaded engine | +14 / −5 |
| `src/vjaudit/generate/pin.py` | also pins judges (`model:` entries that have a `revision:` line) | +7 / −3 |
| `src/vjaudit/generate/runner.py` | frame size becomes part of the generation ID when it is not the default 768 px | +15 / −5 |
| `src/vjaudit/judge/capture.py` | `optional` stages and `select_stages`; `judge_rev` is the revision for local judges; report title "Judge capture report" | +15 / −3 |
| `src/vjaudit/schemas.py` | new record types `Unit` and `Label` | +25 / −0 |
| `tests/data/test_cli_data.py` | updated for the new config paths and per-profile folders, plus new cases | +24 / −15 |
| `tests/generate/conftest.py` | updated for the new config paths and per-profile folders, plus new cases | +1 / −1 |
| `tests/generate/test_cli_gen.py` | updated for the new config paths and per-profile folders, plus new cases | +21 / −3 |
| `tests/generate/test_coverage.py` | updated for the new config paths and per-profile folders, plus new cases | +3 / −3 |
| `tests/generate/test_pin.py` | updated for the new config paths and per-profile folders, plus new cases | +14 / −1 |
| `tests/generate/test_runner.py` | updated for the new config paths and per-profile folders, plus new cases | +19 / −0 |
| `tests/generate/test_smoke_gpu.py` | updated for the new config paths and per-profile folders, plus new cases | +1 / −1 |
| `tests/judge/test_capture.py` | updated for the new config paths and per-profile folders, plus new cases | +1 / −1 |
| `tests/judge/test_capture_cli.py` | updated for the new config paths and per-profile folders, plus new cases | +1 / −1 |
| `tests/judge/test_cli_judge.py` | updated for the new config paths and per-profile folders, plus new cases | +26 / −0 |
| `tests/scripts/test_download_scripts.py` | updated for the new config paths and per-profile folders, plus new cases | +25 / −0 |
| `tests/test_config.py` | updated for the new config paths and per-profile folders, plus new cases | +3 / −3 |
| `tests/test_e2e.py` | updated for the new config paths and per-profile folders, plus new cases | +3 / −3 |
| `uv.lock` | locks the new dependencies and Python 3.11–3.13 (torch 2.10.0, transformers 4.57.6, vLLM 0.18.1 unchanged) | +835 / −249 |

### 3.4 Removed

None. (The three config files listed under 3.2 moved; nothing was deleted.)
