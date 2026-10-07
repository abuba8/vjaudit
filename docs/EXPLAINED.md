# vjaudit, Explained

This document explains the project from the ground up: the research idea, what the code does, and how to
run everything on a Mac. It assumes no prior knowledge of the codebase. Read it top to bottom once; after
that, use the table of contents to jump to what you need.

- **Parts 1–2** explain the idea.
- **Parts 3–4** explain the code.
- **Parts 5–7** show how to run it on your Mac, label with Claude, and read the results.
- **Parts 8–10** cover limits, troubleshooting and a glossary.

`docs/GUIDE.md` is the companion reference for the GPU server and the file-by-file details.

## Contents

1. [The idea](#1-the-idea)
2. [A worked example](#2-a-worked-example)
3. [The pipeline in eight steps](#3-the-pipeline-in-eight-steps)
4. [A tour of the code](#4-a-tour-of-the-code)
5. [Running it on your Mac](#5-running-it-on-your-mac)
6. [Labelling with Claude](#6-labelling-with-claude)
7. [Reading the report](#7-reading-the-report)
8. [What the Mac run can and cannot show](#8-what-the-mac-run-can-and-cannot-show)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

---

## 1. The idea

### 1.1 The problem

In **open-ended video question answering** (VideoQA), a model watches a short video and answers a question
in its own words. For example:

> **Question:** what is the man slicing?
> **Model's answer:** "He is cutting a red onion."

A free-text answer cannot be checked with an exact string match, so benchmarks use a **judge**: a language
model that reads the question, the benchmark's **reference answer** ("onion") and the model's answer, and
says *yes* (correct) or *no* (wrong). A model's **accuracy** is the share of its answers that the judge
calls correct. That number is what papers report and compare.

The catch: that number measures three things at once.

1. **The model** — did it answer correctly?
2. **The reference answer** — is the benchmark's answer itself right and complete?
3. **The judge** — did it compare the two answers correctly?

Papers treat the number as if it measured only the first. This project measures how much the other two
contribute, and whether that changes the conclusions people draw.

### 1.2 Three ways a score goes wrong

| Error | What happens | Example |
|---|---|---|
| **Reference error** | The benchmark's answer is wrong, ambiguous, or only one of several valid answers. | The video shows a man slicing a *shallot*; the reference says "onion". A correct answer is marked wrong. |
| **Matching error** | The judge compares the two texts badly. | It accepts "a vegetable" for "onion" (too general), or rejects "a red onion" because the words differ. |
| **Perception error** | A judge that also watches the video misreads it. | It sees a knife and an onion but decides the man is peeling, not slicing. |

The Mac judges read text only, so they can make the first two kinds of error. Judges that watch the
video, and so can make the third, come in a later stage of the project.

### 1.3 The twist: the errors depend on the model being graded

If a judge made the same mistakes for every model, rankings would stay fair: every score would be off by
the same amount. The project tests whether that is true. There are good reasons to expect it is not:

- **Blind guessing.** A model that does not look at the video can still guess well from the question
  ("what is the man slicing?" → "onion" is a good guess). A lenient judge may reward such guesses.
- **Verbosity.** Long answers contain more words that overlap with the reference, and judges may favour
  them.
- **Hallucinated detail.** "A man slicing an onion with a golden knife in Paris" adds details nobody can
  check. Judges may ignore them or punish them unevenly.

If error rates differ between models, then:

- comparing two models through the judge can give the wrong winner;
- "average" fixes, such as one global correction for all models, do not repair the comparison;
- conclusions about vision itself become distorted. The clearest case is the **visual gain**: how much a
  model improves when it actually sees the video, compared with answering blind.

### 1.4 What the full project will test

These are the main hypotheses from `RESEARCH_DIRECTION.md`, in plain words:

| ID | Plain-language claim |
|---|---|
| **HA** | Reference answers are often wrong or incomplete, especially on automatically generated benchmarks such as MSVD-QA. |
| **HB** | Modern text judges compare answers almost as well as people do; the old default judge (GPT-3.5) does not. |
| **HC** | For a fixed judge, the rate of wrongly accepted answers differs between models; blind and verbose answers are accepted wrongly more often. |
| **HD** | The judge-measured visual gain differs from the visual gain people would measure. |
| **HE** | Judges that watch the video fix some reference errors but add perception errors of their own. |
| **HG** | A protocol with a small amount of human labelling per model keeps comparisons valid, and old GPT-3.5 numbers cannot be converted to the new scale with one formula. |

The full project needs people to label answers, both against the reference and against the video. The
Mac version in this repository is a smaller, earlier step (see 1.6).

### 1.5 Why GPT-3.5 matters

For years the community's default judge has been `gpt-3.5-turbo`, using the Video-ChatGPT prompt. OpenAI
shuts it down on **2026-10-23**. After that date, nobody can ask it to grade new answers again, so the
pipeline can capture its verdicts before the shutdown. Capturing needs an OpenAI API key. On the Mac,
those stages (`G1`, `G2`) are optional and off by default.

### 1.6 What the Mac version does

The Mac version runs the *whole pipeline* end to end on a 32 GB Apple-Silicon laptop, with small models,
so that every step can be tested and inspected before the expensive runs:

- **Three small video models** answer the questions: Qwen2.5-VL-3B, Qwen3-VL-2B and LLaVA-OneVision-0.5B.
- **Two small open judges** grade the answers: Qwen3-4B (main judge) and Qwen2.5-1.5B (small judge).
  They run locally, cost nothing and need no API key.
- **Claude labels a subset by hand.** You give Claude ready-made files; Claude labels each answer
  against the reference. Claude's labels act as a *stronger reference judge* to compare the small judges
  against. They are never called human labels.
- **A report** compares the judges with Claude's labels and shows a first visual-gain estimate.

It runs at two sizes:

| Run | Questions per benchmark | Questions in total | Answers (5 systems) |
|---|---|---|---|
| Run 1 | 100 | 300 | 1,500 |
| Run 2 | 1,000 | 3,000 | 15,000 (Run 1's 1,500 are reused) |

---

## 2. A worked example

Let's follow one question through the whole pipeline.

**The question** comes from MSVD-QA:

> **Question:** what is the man slicing? **Reference answer:** onion

**Step 1: three systems answer.** A *system* is a model plus a way of asking it. `qwen25vl3b-blind`
is the same Qwen model answering without the video.

| System | Raw answer | Normalised answer |
|---|---|---|
| `qwen25vl3b` | "An onion." | `onion` |
| `qwen25vl3b-blind` | "A tomato." | `tomato` |
| `llavaov05b` | "onion" | `onion` |

*Normalising* lower-cases the text, removes punctuation and drops a leading "a/an/the". Two answers with
the same normalised form are the same answer.

**Step 2: the judges grade every answer.** Each judge is asked twice, with two prompt templates:

- **T-VCG**, the original Video-ChatGPT prompt. It asks for a reply like `{'pred': 'yes', 'score': 4}`.
- **T-RUB**, a rubric prompt. It asks for a single word: `Yes` or `No`.

Besides the verdict, the code records the judge's **margin**: how much more likely the judge found "yes"
than "no". It is computed from the judge's token probabilities:

> margin = log P(yes) − log P(no)

A margin of +3 is a confident yes (about 95% yes), +0.2 a hesitant yes (about 55%), and −2 a fairly
confident no (about 12% yes). Margins let later analyses move the yes/no threshold without asking the
judge again.

| System | Qwen3-4B, T-RUB | margin |
|---|---|---|
| `qwen25vl3b` | Yes | +4.1 |
| `qwen25vl3b-blind` | No | −3.2 |
| `llavaov05b` | Yes | +4.1 |

**Step 3: Claude labels each *distinct* answer once.** The two "onion" answers form a single **unit**, so
Claude sees two units for this question, not three, and never sees which model wrote what:

| Unit | Candidate answer | Claude's label |
|---|---|---|
| `3f9a…` | An onion. | correct |
| `b71c…` | A tomato. | incorrect |

**Step 4: the report compares them.** Over all 100 MSVD questions, the report might show that Qwen3-4B
gives `qwen25vl3b` an accuracy of 0.62, while Claude's labels give 0.55. The judge is then 7 points too
generous for that system. If it is 2 points too generous for another system, the comparison between the
two systems is distorted. Spotting that kind of difference is the point of the project.

---

## 3. The pipeline in eight steps

```
 datasets ──► 1 load + check ──► 2 sample ──► 3 frames ──► 4 answers ──► 5 judging ──► 6 labels ──► 7 report
                (items)          (subsets)    (cached)    (generations)  (judge calls)  (units, labels)
                                                                                        8 freeze (archive)
```

Every step reads its inputs from the **artifact store** and writes its outputs back to it (Part 4.2). Every
step can be stopped and re-run: finished work is detected and skipped.

### Step 1: load and check the data

- **What:** read the three benchmarks' question files and find each question's video. Then check that
  the video really opens and decodes.
- **Why:** a missing or broken video would silently bias the sample.
- **Where:** `src/vjaudit/data/loaders/` (one loader per benchmark), `src/vjaudit/data/availability.py`.
- **Output:** one `item_pool` record per question, plus `docs/data_availability_mac.md`.

| Benchmark | What it is | Questions in the test split |
|---|---|---|
| `msvd` | MSVD-QA: short clips, questions generated automatically from captions | 13,157 |
| `anetqa` | ActivityNet-QA: longer videos, questions written by people | 8,000 |
| `nextoe` | NExT-QA open-ended: causal and temporal questions | 9,178 |

### Step 2: draw the samples

- **What:** pick questions at random (fixed seed 42), stratified by question type. The sets are nested:
  `main` (1,000 per benchmark) contains `annotated` (100), which contains `vid` (30). The annotated set is
  also split 40/60 into `dev` and `test`, for later calibration experiments.
- **Why:** nesting means Run 2 reuses everything Run 1 computed. A fixed seed means the same samples
  every time.
- **Where:** `src/vjaudit/data/sampling.py`.
- **Output:** one `item` record per question, plus `docs/samples_mac.md`.

### Step 3: extract frames

- **What:** take 8 evenly spaced frames from each video and shrink them so the longest side is 448
  pixels. Save them to a cache.
- **Why:** models see a handful of frames, not the whole video. Caching means each video is decoded only
  once.
- **Where:** `src/vjaudit/data/frames.py`. The settings are in `configs/profiles/mac/`.

### Step 4: generate answers

- **What:** every system answers every question. The Mac has five systems:

| System | Model | How it is asked |
|---|---|---|
| `qwen25vl3b` | Qwen2.5-VL-3B | with video, short answer |
| `qwen25vl3b-sentence` | Qwen2.5-VL-3B | with video, one full sentence plus a reason |
| `qwen25vl3b-blind` | Qwen2.5-VL-3B | **no video**, short answer |
| `qwen3vl2b` | Qwen3-VL-2B | with video, short answer |
| `llavaov05b` | LLaVA-OneVision-0.5B | with video, short answer |

- **Why:** the blind and sentence variants are there on purpose. They create the "blind guessing" and
  "verbosity" conditions from Part 1.3.
- **Where:** `src/vjaudit/generate/` (Part 4.4).
- **Output:** one `generation` record per (question, system).

### Step 5: judge the answers

- **What:** each judge grades each answer with each template, in a fixed order of **stages**:

| Stage | Set | Judge | Template | Run |
|---|---|---|---|---|
| M1 | annotated | Qwen3-4B | T-VCG | 1 |
| M2 | annotated | Qwen3-4B | T-RUB | 1 |
| M3 | annotated | Qwen2.5-1.5B | T-VCG | 1 |
| M4 | annotated | Qwen2.5-1.5B | T-RUB | 1 |
| M5 | main | Qwen3-4B | T-VCG | 2 |
| M6 | main | Qwen3-4B | T-RUB | 2 |
| G1, G2 | annotated | GPT-3.5 | T-VCG, T-RUB | optional (needs an OpenAI key) |

- **Why:** two judges of different sizes and two templates show how much the score depends on the
  judge set-up, not on the model being graded.
- **Where:** `src/vjaudit/judge/` (Part 4.5).
- **Output:** one `judgecall` record per (answer, judge, template). Each holds the verdict, the margin,
  the top alternative tokens and the raw reply text.

### Step 6: label with Claude

- **What:** group answers into units, export them as ready-to-paste files, give the files to Claude,
  and import Claude's replies.
- **Why:** the small judges need something better to be compared against.
- **Where:** `src/vjaudit/annotate/`. Part 6 is the step-by-step guide.
- **Output:** `unit` and `label` records.

### Step 7: report

- **What:** accuracy per system under each judge set-up and under Claude's labels. Then agreement
  numbers and the visual gain.
- **Where:** `src/vjaudit/report/`.
- **Output:** `docs/results_mac.md` (Part 7).

### Step 8: freeze (optional on the Mac)

- **What:** export every record in a fixed order, with checksums, as an archive.
- **Where:** `src/vjaudit/freeze.py`, `scripts/freeze_stage1.py`.

---

## 4. A tour of the code

### 4.1 Folder map

```
vjaudit/
├── Makefile                  one-line commands for every step (make <target> PROFILE=mac)
├── configs/
│   ├── profiles/mac/         the Mac set-up: generators.yaml, judges.yaml, grid.yaml
│   ├── profiles/server/      the GPU-server set-up (same three files)
│   ├── benchmarks.yaml       where each benchmark's files live and what they are called
│   ├── paths.example.yaml    template for configs/local/paths.yaml (your machine's folders)
│   └── prompts/              generation prompts and judge prompts (T-VCG, T-RUB)
├── src/vjaudit/              the Python package (below)
├── scripts/download/         dataset download scripts
├── tests/                    CPU tests (make sanity)
└── docs/                     this file, GUIDE.md, decisions.md, generated reports
```

### 4.2 The foundations

**Profiles** (`src/vjaudit/profile.py`). A profile is a folder of three config files for one kind of
machine. `load_profile("mac")` reads `configs/profiles/mac/` and returns a `Profile` object with the
models, the judges, the sample sizes and the stage list. Each profile writes to its own folders
(`artifacts/mac/`, `artifacts/frames/mac/`, `artifacts/api_cache/mac/`), so Mac and server results never
mix. Every command that uses a profile prints `profile: <name>` first, so you always know which one is
active.

**Config models** (`src/vjaudit/config.py`). Every YAML file is checked against a strict schema when it is
loaded, so a typo in a config fails immediately with a clear message instead of halfway through a run.

**Records** (`src/vjaudit/schemas.py`). Everything the pipeline produces is one of these record types:

| Record | One per | Key fields |
|---|---|---|
| `Item` | question | question, reference answers, video path, which subsets it belongs to |
| `Generation` | (question, system) | the answer text, its normalised form, which frames were used |
| `JudgeCall` | (answer, judge, template) | verdict, margin, top tokens, parse status, raw reply |
| `Unit` | (question, distinct answer) | the answer shown to the labeller, which systems gave it (hidden from labellers) |
| `Label` | (unit, label source) | correct / incorrect, error tags, note, source (`claude-manual`, `human:<id>`) |

Every record also carries its provenance: the hash of the config that produced it, the git commit and a
timestamp.

**The artifact store** (`src/vjaudit/store.py`). A folder of append-only JSON-lines files, one per record
kind. Each record has a **content-derived ID**. For example, a generation's ID is a hash of the question,
the system and the full generation config. Two consequences:

1. **Re-running is safe.** Work whose ID already exists is skipped. Stop any command with Ctrl-C and
   run it again; it continues where it stopped.
2. **Changing a setting never overwrites old results.** Change the number of frames and the IDs change,
   so new records are added next to the old ones.

**Answer normalisation** (`src/vjaudit/normalize.py`). Lower-case, punctuation removed, one leading
article dropped. It decides which answers count as "the same" for units.

**Device choice** (`src/vjaudit/device.py`). Uses the Apple GPU (`mps`) if present, then CUDA, then CPU,
with half precision (float16) on the Apple GPU.

### 4.3 Data (`src/vjaudit/data/`)

| File | Job |
|---|---|
| `loaders/msvd.py`, `loaders/anetqa.py`, `loaders/nextqa.py` | read each benchmark's raw files into `Item` records |
| `availability.py` | find each item's video file and check it decodes |
| `sampling.py` | the nested, stratified samples and the dev/test split |
| `frames.py` | evenly spaced frame extraction, resizing, caching |

### 4.4 Generation (`src/vjaudit/generate/`)

| File | Job |
|---|---|
| `systems.py` | turns each model config into systems (`qwen25vl3b`, `-sentence`, `-blind`) |
| `adapters.py` | builds the chat message for each model family (video placeholder + question) |
| `hf_engine.py` | **the Mac engine**: runs a model with Hugging Face `transformers` on the Apple GPU |
| `engine.py` | the GPU-server engine (vLLM) and a fake engine for tests |
| `factory.py` | picks the engine from the config's `backend` (`hf` → Mac engine, `vllm` → server engine) |
| `runner.py` | answers a list of questions with one system and stores the generations |
| `coverage.py` | runs every system on a subset; loads one model at a time and frees it before the next |
| `smoke.py` | the quick check behind `make smoke` |
| `pin.py` | writes each model's exact Hugging Face revision into the profile (reproducibility) |

How one Mac answer is produced:

1. `adapters.py` builds the messages: a video placeholder followed by the question and the instruction
   from `configs/prompts/generation/short.txt` ("Answer the question using a single word or phrase.").
2. The model's own processor turns the messages into the exact prompt text.
3. The same processor converts the 8 cached frames into model input.
4. The model generates greedily (no randomness), up to 128 new tokens for short answers.
5. The new tokens are decoded into text and stored as a `Generation`.

### 4.5 Judging (`src/vjaudit/judge/`)

| File | Job |
|---|---|
| `templates.py` | loads T-VCG and T-RUB; fills in question, reference and answer safely |
| `local.py` | **the Mac judge**: runs a local model and returns an OpenAI-shaped reply |
| `api_openai.py` | the judge client: caching, retries, budget; works for both API and local judges |
| `factory.py` | builds the right client for each judge (`backend: transformers` → local) |
| `capture.py` | runs the stages in order and stores `JudgeCall` records |
| `parse.py` | reads the verdict from the reply; an unreadable reply is marked, never guessed |
| `readout.py` | computes the yes/no margin from the token probabilities |
| `smoke.py` | the quick check behind `make smoke` |

**The key trick.** The local judge (`LocalTransport` in `local.py`) answers in exactly the same format
as OpenAI's API, including the probabilities of the top 20 alternative tokens at each position. So
everything downstream works unchanged for GPT-3.5 and for the Mac judges: caching, parsing, margins and
reports.

**Margins and spelling.** A tokenizer may spell "yes" as `Yes`, ` Yes` (with a leading space) or `yes`.
The margin adds up the probabilities of all spellings of yes, and of all spellings of no, before
comparing them.

### 4.6 Labelling (`src/vjaudit/annotate/`)

| File | Job |
|---|---|
| `units.py` | groups answers into units (one per question and distinct answer); plans batches without splitting a question |
| `export.py` | writes `batch_NN.md` files for Claude, or `batch_NN.csv` sheets for people, plus `manifest.json` |
| `importer.py` | reads Claude's reply or a filled sheet, checks every row, stores labels only if all rows are valid |

### 4.7 Reporting (`src/vjaudit/report/`)

| File | Job |
|---|---|
| `metrics.py` | Cohen's κ, false-positive and false-negative rates, yes-rate |
| `summary.py` | builds the result tables and writes `docs/results_<profile>.md` |

### 4.8 The command line

Every step is a command: `uv run python -m vjaudit.cli <group> <command>`. The Makefile wraps the common
ones. The groups are:

| Group | File | Commands |
|---|---|---|
| `data` | `src/vjaudit/cli_data.py` | `check-availability`, `sample` |
| `gen` | `src/vjaudit/cli_gen.py` | `pin-revisions`, `smoke`, `run`, `coverage`, `rerun-check`, `mc-report` |
| `judge` | `src/vjaudit/cli_judge.py` | `smoke`, `capture`, `verify-vcg`, `vendor-rub` |
| `annotate` | `src/vjaudit/cli_annotate.py` | `export`, `import`, `status` |
| `report` | `src/vjaudit/cli_report.py` | `summary` |

Add `--help` to any command to see its options, for example
`uv run python -m vjaudit.cli annotate import --help`.

---

## 5. Running it on your Mac

Run every command from the repository folder (`vjaudit/`). Add `PROFILE=mac` to every `make` command.
Without it, the command uses the GPU-server profile; you will notice, because the first line of output
says `profile: server`.

> **Shortcut:** run `export PROFILE=mac` once in your terminal, and every `make` command in that terminal
> uses the Mac profile.

### What you need

| Item | Amount |
|---|---|
| Memory | 32 GB (only one model is loaded at a time while answering) |
| Free disk | about 30 GB for model downloads (`~/.cache/huggingface`), plus a few GB for frames and results |
| Datasets | already downloaded under `data/` with the scripts in `scripts/download/` |
| Internet | for the one-time model downloads and `make pin` |
| Tools | `uv` (https://docs.astral.sh/uv/) and `git` |

### Step 0: install

```bash
uv python install 3.11
uv sync --extra mac          # installs torch, transformers and friends (Apple-GPU builds)
make sanity                  # CPU tests; must end with "passed" and no "failed"
```

### Step 1: check your folder settings

`configs/local/paths.yaml` tells the code where the data lives. If it does not exist, create it:

```bash
cp configs/paths.example.yaml configs/local/paths.yaml
```

The default values match the download scripts, so usually nothing needs editing.

### Step 2: check the data

```bash
make data-check PROFILE=mac
```

This finds and test-decodes every video, so it takes a while (tens of minutes). It ends with a line like
this one (your numbers will differ slightly):

```
30335 items, 30102 available; report: docs/data_availability_mac.md
```

Open `docs/data_availability_mac.md` to see which videos are missing, if any. A `WARNING` line means a
benchmark has fewer usable questions than the sample needs.

### Step 3: draw the samples

```bash
make data-sample PROFILE=mac
```

Expected: `wrote ... items; report: docs/samples_mac.md`. The report shows 1,000 main, 100 annotated and
30 vid questions per benchmark.

Run this only once. If you run it again, it refuses to overwrite the existing sample. To start over on
purpose, run `uv run python -m vjaudit.cli data sample --force --profile mac`; it archives the old sample
first. Do this before generating answers, not after: a new sample means new questions.

### Step 4: pin the model versions

```bash
make pin PROFILE=mac
git commit -am "pin mac model revisions"
```

This looks up the current version (a commit SHA) of each model on the Hugging Face Hub and writes it into
`configs/profiles/mac/generators.yaml` and `judges.yaml`. From then on, the same version is used every
time, which makes results reproducible. The output lists one SHA per model:

```
profile: mac
qwen25vl3b: <40-character commit id>
qwen3vl2b: <40-character commit id>
...
qwen25-1.5b: <40-character commit id>
```

### Step 5: smoke test (about 10 minutes, plus downloads)

```bash
make smoke PROFILE=mac
```

The first time, this downloads all five models (about 25 GB). Then each video model answers two
questions, with and without the video, and each judge grades one fixed example. You should see two
tables like these (the numbers are only illustrations):

```
| generator | item | mode | seconds | device | answer |
|---|---|---|---|---|---|
| qwen25vl3b | anetqa:test:v_... | video | 3.1 | mps | playing guitar |
| qwen25vl3b | anetqa:test:v_... | blind | 0.8 | mps | dancing |
...
| judge | template | verdict | margin | seconds | top tokens |
|---|---|---|---|---|---|
| qwen3-4b | t-vcg | True | 6.214 | 2.4 | 'yes':-0.00, 'no':-6.21, 'Yes':-9.50 |
| qwen3-4b | t-rub | True | 9.870 | 0.4 | 'Yes':-0.00, 'No':-9.87, 'yes':-12.10 |
```

What to check:

- `device` says `mps` (the Apple GPU). `cpu` means torch cannot see the GPU, and everything will be slow.
- Answers are short, sensible phrases. Empty or garbled answers are a problem (see Part 9).
- Each judge gives a verdict and a margin. A margin of `—` means the judge's first token was not yes/no.
- Any `ERROR:` row names the failing model and the reason.
- The `seconds` column tells you how long the real runs will take: multiply by the number of answers in
  the table at the end of this part.

### Step 6: verify the judge prompts

The judging step refuses to use a prompt whose origin has not been checked. The check makes sure that
T-VCG is word-for-word the official Video-ChatGPT prompt:

```bash
bash scripts/download/fetch_upstream_prompts.sh
git commit -am "verify T-VCG prompt"      # the script marks it verified in configs/prompts/judge/source.yaml
```

T-RUB comes from the earlier BlackboxNLP pipeline. If you have that repository, replace the shipped
reconstruction with the real template (`docs/GUIDE.md`, "Verify the judge prompts"). If you do not, add
`ALLOW_UNVERIFIED=1` to the judge commands below. That is fine for testing the pipeline, but note in your
results that T-RUB was a reconstruction.

Do this step **before** any judging. Changing a prompt changes its ID: verdicts made with the old prompt
stay in the store but are left out of the report (the report says how many), and the judging must be run
again.

### Step 7: Run 1 — answers for 100 questions per benchmark

```bash
make run1 PROFILE=mac
```

This answers 300 questions with 5 systems (1,500 answers), loading one model at a time. It ends with a
coverage list:

```
shard 0/1: 1500 new generations
msvd qwen25vl3b missing 0
...
TOTAL missing 0
```

`missing` above 0 means some videos could not be decoded; the log names them. Re-running is safe and
only fills the gaps.

### Step 8: Run 1 — judging

```bash
make judge-run1 PROFILE=mac              # add ALLOW_UNVERIFIED=1 if you skipped T-RUB verification
```

This runs stages M1–M4 (6,000 judge calls). It prints the number of jobs per stage, then:

```
captured 6000 calls; 0 failed (re-run to retry them); spent 0.00 USD; completed; report: docs/capture_mac.md
```

`docs/capture_mac.md` lists each stage with its share of readable replies (`ok rate`) and yes-rate.
An ok rate far below 1.0 means the judge often ignores the requested format. That is worth knowing in
itself.

### Step 9: Run 1 — labels from Claude

Follow Part 6. In short:

```bash
make annotate-export PROFILE=mac
# give each annotation/mac/claude/batch_NN.md to Claude; save each reply in annotation/mac/claude/replies/
make annotate-import PROFILE=mac FILES="annotation/mac/claude/replies/batch_01.txt"
make annotate-status PROFILE=mac
```

### Step 10: Run 1 — the report

```bash
make report PROFILE=mac
```

This writes `docs/results_mac.md`. Part 7 explains how to read it.

### Step 11: Run 2 — 1,000 questions per benchmark

Run 2 uses the same commands. Everything Run 1 did is reused automatically.

```bash
make run2 PROFILE=mac          # 13,500 new answers
make judge-run2 PROFILE=mac    # stages M5–M6 with the main judge (27,000 new calls)
make report PROFILE=mac        # adds the Run-2 table to docs/results_mac.md
```

Run 2 is long. Leave it running overnight (keep the Mac awake and plugged in). If it stops, run the same
command again.

### How long and how big

These are rough guides for an M5 with 32 GB; your smoke test gives the real seconds per answer.

| Step | Work | Rough time |
|---|---|---|
| `make run1` | 1,500 answers + first-time frame extraction | 1–2 hours |
| `make judge-run1` | 6,000 judge calls | 1–3 hours |
| Claude labelling | about 9–12 batches of ~100 units | 30–60 minutes of your time |
| `make run2` | 13,500 answers | 8–15 hours |
| `make judge-run2` | 27,000 judge calls | 8–20 hours |

### Stopping, resuming and starting over

- **Stop** any step with Ctrl-C. **Resume** by running the same command again.
- **See what is missing:** `uv run python -m vjaudit.cli gen coverage --subset annotated --profile mac`.
- **Start the Mac results over completely:** move or delete the `artifacts/mac/` folder. The server
  profile's results are not affected.

---

## 6. Labelling with Claude

Claude labels each distinct answer as *correct* or *incorrect* given the question and the reference
answer. Claude does not see the video, the model names or the judges' verdicts.

### 6.1 Export the batches

```bash
make annotate-export PROFILE=mac
```

This writes `annotation/mac/claude/batch_01.md`, `batch_02.md`, and so on, each with about 100 units.
All answers to one question are always in the same batch. Each file is complete on its own: it contains
the task, six labelling rules, nine error tags, the exact reply format and the items. A batch looks like
this:

```markdown
### Q1. what is the man slicing?
Reference answer(s): onion
| unit_id | candidate answer |
|---|---|
| 3f9a1c2e7b5d4a60 | An onion. |
| b71c0d9e2a3f4b81 | A tomato. |
```

If you export again later (for example after more answers were generated), only units that were not
exported before are written, as new batch numbers. Batches you have already labelled stay valid.

### 6.2 Ask Claude

For each batch:

1. Start a **new conversation** with Claude (one batch per conversation keeps replies clean and
   complete).
2. Attach the file `batch_NN.md`, or paste its whole content.
3. Send it. No extra instructions are needed; the file says everything.
4. Claude replies with one CSV block:

   ````text
   ```csv
   unit_id,label,error_tags,note
   3f9a1c2e7b5d4a60,correct,,
   b71c0d9e2a3f4b81,incorrect,,different vegetable
   ```
   ````

5. Copy the whole reply into a text file named after the batch, for example
   `annotation/mac/claude/replies/batch_01.txt` (create the `replies` folder once). Extra sentences
   around the CSV block do no harm; the importer finds the block.

### 6.3 Import the replies

```bash
make annotate-import PROFILE=mac FILES="annotation/mac/claude/replies/batch_01.txt annotation/mac/claude/replies/batch_02.txt"
```

You can import one file at a time or several at once, in any order.

The importer checks every row before storing anything:

- the `unit_id` must be one that was exported;
- the label must be `correct` or `incorrect`;
- error tags must be E1–E9;
- one unit must not appear twice with different labels.

If any row fails, **nothing is stored**, and every problem is listed with its file and row number.
Otherwise you see:

```
stored 98 label(s) from claude-manual; 0 already present (skipped)
  batch_01: 2 unit(s) still missing
```

### 6.4 Check progress

```bash
make annotate-status PROFILE=mac
```

```
| batch | units | labelled | missing |
|---|---|---|---|
| batch_01 | 100 | claude-manual: 98 | 2 |
| batch_02 | 97 | claude-manual: 97 | 0 |
```

### 6.5 When something goes wrong

| Problem | What to do |
|---|---|
| Claude's reply stops early (long batch) | In the first file, delete the last row if it is cut off mid-line. Then, in the same conversation, ask: "Continue from the row after `<last complete unit_id>`, in a new csv block with the same header." Save that reply as a second file and import both. Rows already stored are skipped. |
| A row has an unknown `unit_id` | Claude mistyped an ID. Fix the ID in the text file (copy it from the batch file) and import again. |
| The importer finds no header | The reply has no `unit_id,label,error_tags,note` line. Ask Claude to answer again in the required format. |
| You disagree with some of Claude's labels | Do not edit them in place. Run a second pass under a new source name, for example `SOURCE=claude-manual-v2`. Both sources then appear in the report. |
| You want to label units yourself | Export sheets with `uv run python -m vjaudit.cli annotate export --for human --profile mac`, fill in the `label` column (and tags), then import with `make annotate-import PROFILE=mac SOURCE=human:<your-initials> FILES=...`. A person's labels are kept apart from Claude's. |

### 6.6 The honesty rule

Claude's labels are a **reference judge**: a strong language model reading the same text as the small
judges. They are not human ground truth, and they do not use the video. The report states this at the
top. Label sources must be named `claude-manual…` or `human:<id>`, and the importer refuses anything
else, so the two can never be mixed up.

---

## 7. Reading the report

`make report PROFILE=mac` writes `docs/results_mac.md`, with three sections.

### 7.1 Run 1: accuracy per benchmark

An excerpt (the numbers are only illustrations):

```
### msvd

| system | n | qwen25-1.5b/t-rub/t0 | qwen3-4b/t-rub/t0 | qwen3-4b/t-vcg/t0 | claude-manual |
|---|---|---|---|---|---|
| qwen25vl3b | 100 | 0.710 (0) | 0.640 (0) | 0.660 (2) | 0.580 |
| qwen25vl3b-blind | 100 | 0.520 (0) | 0.410 (0) | 0.430 (1) | 0.330 |
```

- **n** is the number of answers.
- Each **judge column** is named `judge/template/decoding`; `t0` means greedy, temperature 0. The value
  is that judge's accuracy for the system.
- **(k)** counts replies the judge gave that could not be read. They are left out of the accuracy, never
  guessed.
- **[k of n judged]** appears when a judge has graded only some of the answers so far.
- The **`claude-manual` column** is the accuracy according to Claude's labels.

### 7.2 Run 1: agreement with the reference

```
Agreement with `claude-manual` (gap = judge acc − reference acc, in points):

| system | judge config | n | gap (pts) | κ | FPR | FNR |
|---|---|---|---|---|---|---|
| qwen25vl3b | qwen3-4b/t-rub/t0 | 100 | +6.0 | 0.720 | 0.190 | 0.050 |
| qwen25vl3b-blind | qwen3-4b/t-rub/t0 | 100 | +8.0 | 0.650 | 0.150 | 0.090 |
```

| Column | Meaning | How to read it |
|---|---|---|
| **gap (pts)** | judge accuracy minus reference accuracy, in percentage points | **+6.0** means the judge is 6 points more generous than Claude for this system |
| **κ** (Cohen's kappa) | agreement corrected for chance; 1 = perfect, 0 = no better than chance | above 0.8 is very good; 0.6–0.8 good; below 0.4 weak |
| **FPR** (false-positive rate) | of the answers the reference calls *incorrect*, the share the judge accepted | a high FPR is a lenient judge |
| **FNR** (false-negative rate) | of the answers the reference calls *correct*, the share the judge rejected | a high FNR is a strict judge |

**What to look for.** Compare the **gap** and the **FPR** *between systems* for the same judge. If the
gap is +2 for one system and +10 for another, the judge is unfair between them, and a single correction
cannot fix that. That is hypothesis HC in miniature.

### 7.3 Visual gain

```
- msvd / qwen25vl3b: qwen3-4b/t-rub/t0: +0.230; claude-manual: +0.250
```

This is accuracy *with* the video minus accuracy of the same model *without* it, measured once per judge
set-up and once with Claude's labels. If the judge's gain differs clearly from the reference's gain, the
judge distorts how much seeing the video helps. That is hypothesis HD in miniature.

### 7.4 Run 2: accuracy on the main set

The same accuracy table for 1,000 questions per benchmark. Only the main judge (Qwen3-4B) grades the
whole main set; the small judge's columns cover only the annotated part and are marked, for example
`0.640 (0) [100 of 1000 judged]`. There are no label columns: only the annotated set is labelled.

### 7.5 How precise are these numbers?

With 100 answers, an accuracy near 50% has a 95% uncertainty of about **±10 points**. Treat differences of
a few points in Run 1 as noise. With 1,000 answers (Run 2), the uncertainty shrinks to about ±3 points.

---

## 8. What the Mac run can and cannot show

**It can show:**

- that every step works end to end on real data, and resumes cleanly;
- how often small judges ignore the requested format;
- early patterns: whether judge generosity differs between systems, and whether the judge-measured
  visual gain differs from Claude's;
- realistic time and cost estimates for the full runs.

**It cannot show the paper's claims, because:**

- the models are deliberately small; the paper's systems are larger;
- Claude's labels are a reference judge, not people, and they do not use the video. So reference errors
  (HA) and perception errors (HE) cannot be measured;
- T-RUB is a reconstruction unless you vendored the original template (Step 6);
- with 100 questions per benchmark, the uncertainty is about ±10 points (Part 7.5);
- without an OpenAI key, there are no GPT-3.5 verdicts to compare against.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| First line says `profile: server` | You forgot `PROFILE=mac`. Add it, or run `export PROFILE=mac` once. |
| `revision is UNPINNED` | Run `make pin PROFILE=mac` (Step 4). For a quick test only, add `ALLOW_UNPINNED=1`. |
| `t-vcg is not verified` (or `t-rub`) | Do Step 6, or add `ALLOW_UNVERIFIED=1` to test the pipeline. |
| `no sampled items yet` | Run `make data-check PROFILE=mac` and `make data-sample PROFILE=mac` first. |
| `device` shows `cpu` in the smoke test | torch cannot see the Apple GPU. Re-run `uv sync --extra mac` and check that `uv run python -c "import torch; print(torch.backends.mps.is_available())"` prints `True`. |
| `out of memory` | Lower `n_frames` (for example to 6) or `frame_max_side` (for example to 384) in `configs/profiles/mac/`. Changed settings create new records and keep the old ones. |
| Empty or garbled answers from one model | Half precision can be unstable for some models on the Apple GPU. Set `dtype: bfloat16` (or `float32`) for that model in `configs/profiles/mac/generators.yaml` and run again. |
| `ERROR: could not load <model>` in the smoke test | The download failed or the model name is wrong. Check your internet connection and the model name, then run again. |
| One model keeps failing in the smoke test | Remove it from `configs/profiles/mac/generators.yaml`, write down why in `docs/decisions.md`, and continue with the others. |
| A judge stage prints `0 jobs` | Either that stage is already complete, or there are no answers for its set yet (run the generation step first). |
| Judge calls show `failed` | The local model raised an error for those calls. The reason is printed. Re-running retries only the failed calls. |
| Downloads go to the wrong disk | Set `HF_HOME=/path/with/space` before running; models are cached there. |
| `annotate import` lists problems | Nothing was stored. Fix the listed rows (Part 6.5) and import again. |

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Accuracy** | The share of a system's answers called correct, by a judge or by a label source. |
| **Annotated set** | The 100 questions per benchmark that are judged by all judges and labelled. |
| **Artifact store** | The folder of append-only record files (`artifacts/<profile>/`). |
| **Blind system** | A model answering from the question alone, without the video. |
| **Cohen's κ** | Agreement between two labellers, corrected for agreement by chance. |
| **Error tags E1–E9** | Reasons an answer is wrong (too general, partial, extra wrong detail, refusal, …). |
| **FPR / FNR** | False-positive rate (wrong answers accepted) and false-negative rate (right answers rejected). |
| **Frame cache** | Saved frames, so each video is decoded once (`artifacts/frames/<profile>/`). |
| **Generation** | One system's answer to one question. |
| **Judge** | A language model that decides whether an answer matches the reference. |
| **Judge config** | A judge plus a prompt template plus a decoding setting, e.g. `qwen3-4b/t-rub/t0`. |
| **Label source** | Who produced a label: `claude-manual…` (Claude, a reference judge) or `human:<id>` (a person). |
| **Main set** | The 1,000 questions per benchmark used in Run 2. |
| **Margin** | log P(yes) − log P(no) for the judge's verdict; positive = leaning yes. |
| **Normalised answer** | Lower-case, punctuation removed, leading article dropped; used to group identical answers. |
| **Pinning** | Recording the exact model version (commit SHA) so runs are reproducible. |
| **Profile** | A folder of configs for one machine type: `mac` or `server`. |
| **Reference answer** | The benchmark's own answer to a question. |
| **Stage** | One (set, judge, template) block of judging, run in a fixed order (M1–M6). |
| **System** | A model plus a way of asking it, e.g. `qwen25vl3b-blind`. |
| **T-VCG / T-RUB** | The two judge prompts: the original Video-ChatGPT prompt, and a yes/no rubric prompt. |
| **Unit** | One distinct (question, normalised answer) pair; the thing that gets labelled. |
| **Visual gain** | Accuracy with the video minus accuracy without it, for the same model. |
