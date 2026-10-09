# Workflow: how the code runs, and how to run it

This file follows the pipeline one step at a time. For each step it shows the command you type, the code
that runs, what it reads and what it writes. Part 2 lists the exact commands, in order, to run everything
on your Mac with the conda environment `dl`.

For the research idea in plain language, read `docs/EXPLAINED.md`. For the differences from the old
version, read `differences.md`. To run everything from Jupyter instead of the terminal, open
`notebooks/pipeline.ipynb`: the same steps, written out in one self-contained notebook.

---

## Part 1. How the code works, step by step

### 1.1 The big picture

```
 make <target> PROFILE=mac
        │
        ▼
 python -m vjaudit.cli <group> <command>      ← src/vjaudit/cli.py and cli_*.py
        │
        ├─ load the profile                    ← src/vjaudit/profile.py   (configs/profiles/mac/*.yaml)
        ├─ open the artifact store             ← src/vjaudit/store.py     (artifacts/mac/)
        ├─ do the step's work                  ← data/, generate/, judge/, annotate/, report/
        └─ write records + a report            → artifacts/mac/<kind>/   and   docs/*_mac.md
```

Every step follows the same pattern:

1. **Read the profile.** `load_profile("mac")` reads the three files in `configs/profiles/mac/`:
   - `generators.yaml`: the video models;
   - `judges.yaml`: the judge models;
   - `grid.yaml`: sample sizes, frame size and judge stages.

   It also reads `configs/local/paths.yaml`, which says where the data is.
2. **Open the store.** `artifacts/mac/` holds one folder per record kind: `item_pool`, `item`,
   `generation`, `judgecall`, `unit`, `label`. Each folder contains append-only `part-*.jsonl` files.
3. **Skip finished work.** Every record has an ID computed from its inputs, so a step only does the work
   whose ID is not stored yet. That makes every command safe to stop (Ctrl-C) and re-run.
4. **Write new records and a short report.**

The data flows like this:

```
 benchmark files ──► item_pool ──► item ──► generation ──► judgecall ─┐
 (data/…)            (all qs)    (sampled)   (answers)     (verdicts)  ├──► docs/results_mac.md
                                                │                      │
                                                └──► unit ──► label ───┘
                                                   (distinct answers) (Claude's labels)
```

### 1.2 Step 1: find the videos (`make data-check`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli data check-availability` |
| **Code** | `src/vjaudit/cli_data.py` → `check_availability_cmd` |
| **Reads** | `configs/benchmarks.yaml`, the annotation files under `../vjaudit/data/*/annotations`, every video under `../vjaudit/data/*/videos` |
| **Writes** | `artifacts/mac/item_pool/` (one record per question), `docs/data_availability_mac.md` |

What happens:

1. `data/loaders/msvd.py`, `anetqa.py` and `nextqa.py` read each benchmark's question files and turn every
   question into an `Item`: question, reference answers, question type and video id.
2. `data/availability.py` → `index_videos` finds each question's video file by name.
3. `probe_decodable` opens each video with PyAV and decodes a frame at the start and near the end.
   Questions whose video is missing or broken are kept but marked `available: false`.
4. The command prints `N items, M available` and warns if a benchmark has fewer usable questions than the
   sample needs (1,000).

### 1.3 Step 2: draw the samples (`make data-sample`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli data sample` |
| **Code** | `src/vjaudit/cli_data.py` → `sample_cmd`; `src/vjaudit/data/sampling.py` |
| **Reads** | `artifacts/mac/item_pool/`, `sample_sizes` and `seed` from `configs/profiles/mac/grid.yaml` |
| **Writes** | `artifacts/mac/item/` (the sampled questions), `docs/samples_mac.md` |

What happens:

1. `assign_nested` draws nested samples per benchmark, stratified by question type, with seed 42:
   `main` (1,000) contains `annotated` (100), which contains `vid` (30). MSVD gets at least 15 questions per
   type.
2. `split_dev_test` splits the annotated questions 40/60 into `dev` and `test`.
3. Each `Item` is stored with flags `in_main`, `in_annotated`, `in_vid_subset` and `human_split`.

Run 1 uses `in_annotated`; run 2 uses `in_main`. Because the sets are nested, run 2 reuses run 1's work.

### 1.4 Step 3: pin the model versions (`make pin`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli gen pin-revisions` |
| **Code** | `src/vjaudit/cli_gen.py` → `pin_revisions_cmd`; `src/vjaudit/generate/pin.py` |
| **Reads** | the Hugging Face Hub (needs internet) |
| **Writes** | the `revision:` lines in `configs/profiles/mac/generators.yaml` and `judges.yaml` |

`pin_revisions` asks the Hub for each model's current commit id and writes it into the YAML file,
keeping comments. From then on the same model version is downloaded every time. A model left as
`UNPINNED` is refused unless you pass `ALLOW_UNPINNED=1`.

### 1.5 Step 4: smoke test (`make smoke`)

| | |
|---|---|
| **Commands** | `python -m vjaudit.cli gen smoke`, then `python -m vjaudit.cli judge smoke` |
| **Code** | `src/vjaudit/generate/smoke.py`, `src/vjaudit/judge/smoke.py` |
| **Writes** | nothing (it prints two tables) |

`gen smoke` loads each video model in turn and answers 2 sampled questions, once with the video and once
without, then frees the model. The code path is the same as in the real run (section 1.6), so if the smoke
test works, the real run will too.

`judge smoke` loads each local judge and grades one fixed example, `("what is the man doing?", "cooking",
"he is cooking food")`, with both judge prompts. It uses the same path as the real judging (section 1.7).
It shows the verdict, the yes/no margin and the top tokens.

### 1.6 Step 5: generate answers (`make run1`, later `make run2`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli gen run --subset annotated` (run 2: `--subset main`) |
| **Code** | `src/vjaudit/cli_gen.py` → `run_cmd` → `src/vjaudit/generate/coverage.py` → `run_subset` |
| **Reads** | `artifacts/mac/item/`, the videos, `configs/profiles/mac/generators.yaml`, `configs/prompts/generation/*.txt` |
| **Writes** | `artifacts/mac/generation/`, cached frames in `artifacts/frames/mac/` |

What happens, for each video model in `generators.yaml`:

1. **Systems.** `generate/systems.py` → `expand_systems` turns one model into its systems. For example,
   `qwen25vl3b` becomes `qwen25vl3b` (short answer), `qwen25vl3b-sentence` and `qwen25vl3b-blind` (no
   video).
2. **Load the model once.** `generate/factory.py` → `make_engine` builds a `TransformersEngine`
   (`generate/hf_engine.py`), which loads the processor and model onto the Apple GPU (`mps`, float16).
3. **For each question** (`generate/runner.py` → `run_generation`):
   1. `data/frames.py` → `FrameCache.get` takes 8 evenly spaced frames, resizes them to at most 448 px and
      caches them. Blind systems skip this.
   2. `generate/adapters.py` builds the chat message: a video placeholder plus the question and the
      instruction from `configs/prompts/generation/short.txt`. The model's own processor renders it.
   3. `TransformersEngine.generate` runs greedy decoding (no randomness) and returns the answer text.
   4. The answer is normalised (`normalize.py`: lower-case, no punctuation, no leading article) and stored
      as a `Generation`. Its `gen_id` is a hash of the question, the system and the full generation config.
4. **Free the model** (`engine.close()`) before the next model loads, so only one is in memory.

Then `gen coverage` prints, per benchmark and system, how many questions still have no answer.

### 1.7 Step 6: judge the answers (`make judge-run1`, later `make judge-run2`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli judge capture --stages M1,M2,M3,M4 --confirm` (run 2: `M5,M6`) |
| **Code** | `src/vjaudit/cli_judge.py` → `capture_cmd` → `src/vjaudit/judge/capture.py` |
| **Reads** | `artifacts/mac/generation/`, `artifacts/mac/item/`, `capture_stages` in `grid.yaml`, `judges.yaml`, `configs/prompts/judge/*` |
| **Writes** | `artifacts/mac/judgecall/`, the reply cache in `artifacts/api_cache/mac/`, `docs/capture_mac.md` |

What happens:

1. **Stages.** `load_stages` and `select_stages` pick the stages named in `--stages`, in config order.
   Each stage says which set (annotated or main), which judge and which template to use.
2. **Prompt check.** A template whose origin is not verified in `configs/prompts/judge/source.yaml` stops
   the run, unless you pass `ALLOW_UNVERIFIED=1`.
3. **Clients.** `judge/factory.py` → `make_client` builds one client per judge. For a local judge the
   client sends requests to `LocalTransport` (`judge/local.py`). That runs the model with transformers
   and returns an OpenAI-style reply, including the top 20 token probabilities.
4. **Jobs.** `build_jobs` lists every (answer, stage) pair whose judge call is not stored yet.
5. **Judging.** For each job, `run_capture` does the following:
   1. it fills the template (`judge/templates.py` → `render_vcg` or `render_rub`) with the question, the
      first reference answer and the model's answer;
   2. it asks the client, which returns a cached reply if the identical request was made before;
   3. it reads the verdict (`judge/parse.py`); unreadable replies are marked `unparseable` or `off_vocab`,
      never guessed;
   4. it computes the margin, log P(yes) − log P(no), from the token probabilities (`judge/readout.py`);
   5. it stores a `JudgeCall` record.
6. A failed call is counted and listed, and the run continues. Re-running retries only the failed calls.
7. `write_capture_report` writes `docs/capture_mac.md`: per stage, how many calls were expected and done,
   the share of readable replies, and the yes-rate.

### 1.8 Step 7: label with Claude (`make annotate-export`, `make annotate-import`)

| | |
|---|---|
| **Commands** | `python -m vjaudit.cli annotate export --set annotated --for claude`, then `annotate import …`, `annotate status` |
| **Code** | `src/vjaudit/cli_annotate.py`; `src/vjaudit/annotate/units.py`, `export.py`, `importer.py` |
| **Writes** | `artifacts/mac/unit/`, `annotation/mac/claude/batch_NN.md` and `manifest.json`, then `artifacts/mac/label/` |

What happens:

1. **Units.** `build_units` groups the annotated answers by (question, normalised answer). Identical
   answers from different systems become one unit, so Claude labels each distinct answer once.
2. **Batches.** `plan_batches` cuts the units into batches of about 100, never splitting a question.
3. **Export.** `export_claude` writes one self-contained `batch_NN.md` per batch: the task, the rules,
   the error tags, the exact reply format, and the questions with their candidate answers. Model names
   are never shown. `manifest.json` records which units are in which batch.
4. **You** give each file to Claude and save the reply as a text file.
5. **Import.** `parse_label_csv` finds the CSV block in the reply. `import_labels` checks every row (known
   unit id, `correct`/`incorrect`, tags E1–E9), and stores `Label` records only if every row is valid.
   It then reports which units in each batch are still missing.

### 1.9 Step 8: report (`make report`)

| | |
|---|---|
| **Command** | `python -m vjaudit.cli report summary` |
| **Code** | `src/vjaudit/cli_report.py` → `src/vjaudit/report/summary.py`, `report/metrics.py` |
| **Reads** | `item`, `generation`, `judgecall` and `label` records |
| **Writes** | `docs/results_mac.md` |

`build_tables` computes, per benchmark and system:

- the accuracy under each judge set-up (`judge/template/decoding`);
- the accuracy under each label source;
- the agreement with the first label source: the gap in points, Cohen's κ, the false-positive rate and the
  false-negative rate.

`visual_gain` subtracts each system's blind accuracy from its with-video accuracy. Verdicts made with an
older version of a judge prompt are left out, and the report says how many.

### 1.10 Where everything ends up

| Path | What it holds |
|---|---|
| `artifacts/mac/item_pool/`, `item/` | all questions; the sampled questions |
| `artifacts/mac/generation/` | every answer from every system |
| `artifacts/mac/judgecall/` | every judge verdict, margin and raw reply |
| `artifacts/mac/unit/`, `label/` | the units Claude labels; the imported labels |
| `artifacts/frames/mac/` | cached frames (`.npz`), so each video is decoded once |
| `artifacts/api_cache/mac/` | cached judge replies, per judge and model revision |
| `annotation/mac/claude/` | the batch files for Claude, and the folder where you save Claude's replies |
| `docs/data_availability_mac.md`, `docs/samples_mac.md`, `docs/capture_mac.md`, `docs/results_mac.md` | the reports |

---

## Part 2. How to run it: step-by-step commands

Run everything from the `vjauditv2` folder, in a terminal where `dl` is active and `VJAUDIT_PROFILE=mac`
is set (Step 0). Each block says what to
expect. If a step stops (Ctrl-C, a crash or a closed laptop), run the same command again; it continues
where it stopped.

### Step 0: go to the folder and activate the environment (every new terminal)

```bash
cd ~/Documents/Study-ML-Research/cvpr-2027/docs/"Topic C"/vjauditv2
conda activate dl
export VJAUDIT_PROFILE=mac   # make commands and CLI commands now use the Mac set-up
```

### Step 1: install (once)

```bash
python --version             # must say 3.11, 3.12 or 3.13
pip install -e ".[mac]" pytest pytest-asyncio
make sanity                  # expect: "... passed, ... deselected" and no "failed"
```

### Step 2: check the data (once, tens of minutes)

```bash
make data-check
```

Expect a last line like `30335 items, 30102 available; report: docs/data_availability_mac.md`. Open
that report if many videos are missing.

### Step 3: draw the samples (once)

```bash
make data-sample
```

Expect `wrote ... items; report: docs/samples_mac.md`. The report shows 1,000 / 100 / 30 questions per
benchmark.

### Step 4: pin the model versions (once, needs internet)

```bash
make pin
```

Expect one line per model and judge, each with a 40-character commit id.

### Step 5: smoke test (about 10 minutes, plus about 25 GB of downloads the first time)

```bash
make smoke
```

Check that:

- `device` is `mps`;
- the answers are short, sensible phrases;
- each judge shows a verdict and a margin;
- there are no `ERROR:` rows.

The `seconds` column tells you how long the real runs will take.

### Step 6: verify the judge prompts (once, before any judging)

```bash
bash scripts/download/fetch_upstream_prompts.sh
```

This marks T-VCG as verified. If you cannot verify T-RUB (it needs the earlier BlackboxNLP repository,
see `docs/GUIDE.md`), add `ALLOW_UNVERIFIED=1` to the two `judge-run` commands below.

### Step 7: Run 1, answers (100 questions per benchmark, about 1–2 hours)

```bash
make run1
```

Expect `shard 0/1: 1500 new generations`, then a coverage list ending in `TOTAL missing 0`.

### Step 8: Run 1, judging (about 1–3 hours)

```bash
make judge-run1                        # or: make judge-run1 ALLOW_UNVERIFIED=1
```

Expect `captured 6000 calls; 0 failed ...; report: docs/capture_mac.md`.

### Step 9: Run 1, labels from Claude

```bash
make annotate-export
mkdir -p annotation/mac/claude/replies
```

For each `annotation/mac/claude/batch_NN.md`:

1. Open a new Claude conversation.
2. Attach or paste the file, and send it.
3. Save Claude's whole reply as `annotation/mac/claude/replies/batch_NN.txt`.

Then import the replies and check what is missing:

```bash
make annotate-import FILES="annotation/mac/claude/replies/batch_01.txt annotation/mac/claude/replies/batch_02.txt"
make annotate-status
```

The import stores nothing if any row is wrong, and lists each problem with its file and row number. Fix
those rows and run the import again. Repeat until `annotate-status` shows `missing` 0 everywhere.

### Step 10: Run 1, report

```bash
make report
open docs/results_mac.md
```

### Step 11: Run 2 (1,000 questions per benchmark; leave it overnight)

```bash
make run2                              # 13,500 new answers
make judge-run2                        # 27,000 new judge calls (add ALLOW_UNVERIFIED=1 if you used it before)
make report                            # adds the run-2 table to docs/results_mac.md
```

Keep the Mac plugged in and awake. If anything stops, run the same command again.

### Cheat sheet

| What | Command |
|---|---|
| Start a session | `cd …/vjauditv2 && conda activate dl && export VJAUDIT_PROFILE=mac` |
| Tests | `make sanity` |
| Data | `make data-check`, then `make data-sample` |
| Models | `make pin`, then `make smoke` |
| Run 1 | `make run1`, then `make judge-run1`, then labelling, then `make report` |
| Labelling | `make annotate-export`, then `make annotate-import FILES="…"`, then `make annotate-status` |
| Run 2 | `make run2`, then `make judge-run2`, then `make report` |
| What is missing | `python -m vjaudit.cli gen coverage --subset annotated` |
| Help for any command | `python -m vjaudit.cli <group> <command> --help` |

If something goes wrong, see the troubleshooting table in `docs/EXPLAINED.md` (Part 9).
