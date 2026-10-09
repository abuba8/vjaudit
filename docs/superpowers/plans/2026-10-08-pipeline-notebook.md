# Pipeline Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `notebooks/pipeline.ipynb`, one self-contained, explanatory notebook that runs the whole pipeline on
the Mac or the server, with tests that execute it and pin its core functions to the package.

**Architecture:**
- The notebook is authored as a percent-format source (`# %%` code / `# %% [markdown]` cells, `# tags: [...]`
  comments) in the scratchpad.
- A small builder converts that source to `.ipynb` with `nbformat`. Only the `.ipynb` ships.
- Cells that only define functions are tagged `lib`, so tests can exec them alone. The settings cell is
  tagged `parameters`.

**Tech Stack:**
- in the notebook: Python ≥ 3.11, numpy, pandas, PyAV, Pillow, PyYAML-free (settings live in the cell),
  tqdm (optional), matplotlib (optional), torch + transformers (Mac), vllm (server), openai (API judges);
- in the tests: nbformat, nbclient, ipykernel.

**Spec:** `docs/superpowers/specs/2026-10-08-pipeline-notebook-design.md` (approved 2026-10-08).

## Global Constraints

- **Self-contained:** no `import vjaudit` / `from vjaudit` anywhere in the notebook. It reads only
  `configs/prompts/**` from the repository.
- **Location:**
  - the notebook runs from `notebooks/`, so `REPO = Path.cwd().parent` (or `NOTEBOOK_REPO` from the
    environment);
  - results go to `notebooks/runs/<profile>[-dryrun]/`;
  - `DATA_ROOT=None` means `REPO.parent/"vjaudit"/"data"` on the Mac and `REPO/"data"` on the server.
- **Settings cell** (tag `parameters`): `PROFILE`, `DRY_RUN`, `DATA_ROOT`, `RUN`, `RUN_LIMIT`,
  `RUN_STAGES`, `REBUILD_POOL`, `REBUILD_SAMPLE`, plus the `MAC` and `SERVER` dictionaries with the
  spec §4 values.
- **Dry run:** samples 3/3/1, `fake` generators and judges, results in `runs/<profile>-dryrun/`.
- **IDs:** as in spec §5; prompt text and frame settings are part of the IDs.
- **Decoding:** greedy decoding overrides `do_sample=False, temperature=None, top_p=None, top_k=None,
  repetition_penalty=1.0`; `do_sample_frames=False` with videos.
- **Unparseable verdicts** are counted, never guessed. Label sources must match
  `^(claude-manual(-[A-Za-z0-9_.]+)?|human:[A-Za-z0-9_.-]+)$`.
- **Saved without outputs.** Every section's markdown has *What / Why / Reads → writes / Mac vs server*.

## Review Focus

1. **The kernel restarts in the middle of section 8 or 9.** Re-running the section continues; stored rows are
   not duplicated. *(Task 3 test `test_generation_resumes_without_duplicates`.)*
2. **A section is run after a restart without running the earlier sections.** Each section reloads its
   inputs from the store, and a missing input gives a clear "run section N first" message.
   *(Task 3 test `test_section_needs_previous_output_message`.)*
3. **`DATA_ROOT` is wrong.** Section 4 stops with a message naming the folder it looked in.
   *(Task 2 test `test_missing_data_root_message`.)*
4. **The Claude reply is messy** (prose, CRLF, quoted notes). The import extracts the CSV block.
   *(Task 5 equivalence test on `parse_label_csv`.)*
5. **Judge prompt files change after judging.** New verdict IDs are created, and the report counts only
   current verdicts. *(Task 4 test `test_template_change_gives_new_ids`.)*

---

### Task 1: Scaffold, settings, helpers, test harness

**Files:** create the notebook source (sections 0–3), `notebooks/README.md`, the builder script
(scratchpad), and `tests/notebook/{__init__,conftest,test_hygiene}.py`. Modify `.gitignore`
(`/notebooks/runs/`) and `pyproject.toml` (dev group: `nbformat`, `nbclient`, `ipykernel`; extra
`notebook`: `jupyterlab`, `ipykernel`, `tqdm`, `matplotlib`; marker list unchanged).

**Interfaces (notebook `lib` cells):**
- `settings() -> dict`, the resolved settings: `profile`, `dry_run`, `data_root`, `runs`, `run`,
  `run_limit`, `stages`, `generators`, `judges`, `samples`, `frames`, `frame_max_side`, `seed`,
  `msvd_min_per`, `frac_dev`.
- `Store(root: Path)`:
  - `.load(table) -> pd.DataFrame`;
  - `.keys(table) -> set[str]`;
  - `.append(table, rows: list[dict], key: str) -> int`;
  - `.rebuild(table)`, which renames the file with a timestamp.
- `short_id(*parts) -> str` (sha1, 16 hex characters), `normalize_answer(text) -> str`,
  `need(df, section)`, which raises `RuntimeError("run section N first")` when `df` is empty.

**Tests:**
- `tests/notebook/conftest.py`:
  - `load_lib(params: dict | None = None) -> dict` execs the `parameters` cell (with overrides) and all
    `lib` cells into a namespace;
  - `run_notebook(params: dict, cwd: Path) -> nbformat.NotebookNode` executes it with nbclient.
- `test_hygiene.py`: no `vjaudit` import; no outputs saved; every `## ` section header appears in the
  section 0 table of contents; the `parameters` cell is tagged.
- Equivalence: `normalize_answer` matches the package on a list of tricky strings.

### Task 2: Data — load, check videos, sample (sections 4–6)

**Interfaces:**
- `load_benchmarks(data_root) -> pd.DataFrame`. Its columns are `item_id, benchmark, split, video_id,
  question, answers_ref, qtype, mc_options, mc_answer_idx`, with loaders `load_msvd`, `load_anetqa` and
  `load_nextoe` mirroring the package.
- `check_videos(df, data_root, probe=probe_decodable) -> pd.DataFrame` adds `video_path, available, problem`.
- `assign_nested(pool, sizes, seed, msvd_min_per) -> pd.DataFrame` and
  `split_dev_test(df, frac_dev, seed) -> pd.DataFrame`. These mirror `allocate`, `stratified_sample` and
  `_vid_ranks` exactly.

**Tests:**
- `test_loaders_match_package`: same item ids, questions, refs, qtypes, MC options on `tests/fixtures/raw`.
- `test_sampling_matches_package`: same `in_main`, `in_annotated`, `in_vid_subset`, `vid_rank` and
  `human_split` on a 60-item synthetic pool, for sizes (40, 20, 10), seed 42 and MSVD minimum 15.
- `test_missing_data_root_message`.

### Task 3: Frames and generation (sections 7–8)

**Interfaces:**
- `sample_indices(n_total, n, start=0, end=None)` mirrors the package;
  `FrameCache(root, max_side).get(video_path, video_id, n, t_start, t_end) -> (np.ndarray, list[int])`.
- `expand_systems(gen: dict, benchmark) -> list[dict]` returns `system_id, generator_id, style, blind,
  n_frames`.
- `build_messages(family, text, blind) -> list[dict]` and `prompt_text(item, style) -> str`.
- Engines, each with `.render(messages) -> str`, `.generate(requests, max_new_tokens) -> list[str]` and
  `.close()`:
  - `FakeEngine(gen)`;
  - `HFEngine(gen, device)`;
  - `VLLMEngine(gen)`.

  `make_engine(gen, dry_run)` chooses between them.
- `run_generation(store, items_df, cfg) -> int` returns the number of new generations.
  `parse_mc_letter(text) -> int | None`.

**Tests:**
- `test_frame_indices_match_package`, `test_systems_match_package` (server generators),
  `test_parse_mc_matches_package`.
- `test_generation_resumes_without_duplicates`: a fake engine that raises after 3 calls, then a re-run.
  Expect no duplicate keys, and all rows present.
- `test_section_needs_previous_output_message`.

### Task 4: Judging (section 9)

**Interfaces:**
- `load_template(tid) -> (parts, hash8)`, `render(tid, question, answer, pred) -> list[dict]`
  (single-pass fill);
- `parse_vcg(text) -> (verdict, score, status)`, `parse_yes_no(text) -> (verdict, status)`;
- `margin_at(content, template) -> (margin, top5)`;
- judges, each with `.complete(messages, max_tokens, temperature, n) -> list[{text, content}]`, plus
  `.close()`:
  - `FakeJudge`;
  - `LocalJudge(cfg)` (transformers, top-20 log-softmax);
  - `APIJudge(cfg, cache_dir)` (OpenAI, cache, 6 retries with backoff, spend cap).
- `run_stages(store, cfg) -> dict` returns a summary per stage: `jobs, done, failed`.

**Tests:**
- `test_render_parse_margin_match_package`;
- `test_api_judge_retries_and_caches`, with a fake `create` function: a 429 then success; a second call
  is a cache hit;
- `test_template_change_gives_new_ids`;
- `test_stage_failures_are_listed_not_fatal`.

### Task 5: Labelling, report, troubleshooting (sections 10–12)

**Interfaces:**
- `build_units(items_df, gens_df) -> pd.DataFrame`, `plan_batches(units, size=100) -> list[DataFrame]`;
- `export_claude(units, items, out_dir) -> list[Path]` (spec §8 exact text), `export_human(...)`;
- `parse_label_csv(text) -> list[dict]`;
- `import_labels(store, rows, source, manifest) -> dict` (`stored, skipped, missing_by_batch`);
  `LabelImportError`;
- `cohen_kappa`, `error_rates`, `build_tables(store, subset) -> dict`, `visual_gain(tables)`,
  `write_results(store, path)`.

**Tests:** equivalence with the package for:
- units and batches;
- the Claude markdown body (identical header text);
- `parse_label_csv` on the messy-reply cases;
- import validation error count;
- κ / FPR / FNR / gap on the hand-made example (κ 0.5, gap 25).

### Task 6: Execution tests, docs, delivery

**Tests:** `tests/notebook/test_execute.py`.
- It builds synthetic data in `tmp_path`: the fixture annotations copied into the three benchmark folders,
  plus generated `.avi`/`.mp4` videos named as the loaders expect.
- It executes the notebook with `PROFILE="mac", DRY_RUN=True, DATA_ROOT=tmp`, then with
  `PROFILE="server", DRY_RUN=True`.
- It asserts:
  - every cell ran;
  - `generations.jsonl` holds the expected count (questions × systems);
  - `judgecalls.jsonl` is non-empty;
  - batch files exist.
- It then writes a synthetic Claude reply into `replies/` and executes the notebook again (finished steps
  are skipped; section 10 imports every reply file it finds), and asserts that `results.md` contains
  "Agreement with `claude-manual`".

**Docs:** a pointer to `notebooks/pipeline.ipynb` in `README.md`, `WORKFLOW.md`, `docs/EXPLAINED.md` (Part 5
intro) and `differences.md` (regenerated). `tests/test_docs.py` also checks `notebooks/README.md`.

**Delivery:** commit; copy the `notebooks/` folder and the changed docs into `Topic C/vjauditv2` (via the
VM clone, as before); verify there with `git status`.
