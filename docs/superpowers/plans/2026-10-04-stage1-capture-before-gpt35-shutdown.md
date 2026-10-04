# Stage 1: Integrity, Data, Generation, and GPT-3.5 Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before OpenAI shuts down `gpt-3.5-turbo` on **2026-10-23**, produce every model answer the study needs and capture GPT-3.5's verdicts and logprobs on all of them, on infrastructure the rest of the project reuses. Settle the prior pipeline's integrity question (SP0) first.

**Architecture:** This is a new Python package, `vjaudit`. Everything is driven by configuration files (YAML) in `configs/`, and every output goes into an append-only JSONL artifact store. Every record is keyed by a content hash, so runs are idempotent and resumable. The pipeline runs in order: data loaders → nested stratified samples → frame cache → generation (vLLM, with family-specific prompt adapters) → GPT-3.5 judge client (cache, budget cap, logprob readout) → capture runner, which works through a fixed priority order so the most valuable judge calls finish first.

**Tech Stack:** Python 3.11, uv, pydantic v2, PyYAML, numpy, PyAV, vLLM (pinned; must support Qwen3-VL), transformers, openai ≥ 1.40, tiktoken, Jinja2, Typer, pytest + pytest-asyncio, GitHub Actions (CPU-only CI).

**Spec:** `docs/Topic C/RESEARCH_DIRECTION.md` (locked 2026-10-03; it decides scope) and `docs/Topic C/REQUIREMENTS_CONTRACT.md` (requirement IDs, data contracts §8, global constraints §7). Where the two conflict, RESEARCH_DIRECTION wins.

**Scope of this plan:** SP0 (all of it), SP1 (except the demonstration pool), SP2 (all of it), and the GPT-3.5 part of SP3. Separate plans will cover the rest (see "Follow-on plans" at the end). **SP4 annotation tooling should be planned and started in parallel with this plan**, because it is the critical path.

**Schedule (hard deadline in bold):**

| Dates (2026) | Tasks |
|---|---|
| Oct 4–6 | 1, 2, 3 |
| Oct 6–9 | 4, 5, 6, 7 |
| Oct 8–11 | 8, then 9a (annotated subset) |
| Oct 9–12 | 10, 11, 12 (CPU only; run in parallel with generation) |
| Oct 12–20 | 13 (capture stages S1→S6); 9b (main sample) runs alongside |
| Oct 21–22 | 14 (freeze and export) |
| **Oct 23** | **GPT-3.5 shut down** |

**Fallback gate:** if the main-sample generations are not finished by Oct 15, still run capture stages S1–S2 and S5 on the annotated subset. Then run S3–S4 on whatever main-sample generations exist, ordered by `item_id`.

## Global Constraints

- Python 3.11. Dependencies pinned in `uv.lock`. vLLM, transformers and torch pinned to exact versions (R-GLB-01).
- Hardware: 3× NVIDIA A30 24 GB. A model that does not fit one GPU in bf16 uses documented AWQ/GPTQ quantization, recorded as an attribute (R-GLB-02).
- Every model is pinned by HF revision SHA or API snapshot string. API calls are cached by request hash, and the cache is a release artifact (R-GLB-03).
- Determinism: greedy decoding for generation. Seeds recorded for all sampling (seed 42 for every sampler in this plan). Rerunning an open-weight config must reproduce outputs exactly on the same hardware and software (R-GLB-04).
- Content-addressed artifacts. Every record carries `config_hash`, `git_sha`, `created_at`. Runs are idempotent and resumable (R-GLB-05).
- Configuration as data. No experiment parameter is hard-coded (R-GLB-06).
- Videos are never redistributed. Releases contain IDs, labels, generations and judge outputs only (R-GLB-08).
- Anonymity: no author names, institution, personal paths or GitHub handles in code, configs or filenames. Machine-specific paths live only in `configs/local/*.yaml`, which is gitignored (R-GLB-09).
- Unit tests for every parser, normalizer, sampler, readout and metric. CI runs on CPU with tiny fixtures and mocked models (R-GLB-10). GPU, API and legacy tests carry the pytest markers `gpu`, `api` and `legacy` and are excluded from CI.
- API spend is capped by `max_usd` in `configs/judges.yaml` and estimated before any run (R-BUD-01).
- Unparseable judge outputs are counted and reported per configuration, never silently mapped to a verdict (§7.1 #10).
- GPT-3.5 capture must be complete by **2026-10-22 23:59 local**. The model shuts down 2026-10-23 (OpenAI deprecations page).
- Generation decoding: greedy, `max_new_tokens=128` (short style) and `256` (sentence style). Uniform frame sampling, 16 frames by default, 32 for ActivityNet-QA where memory allows (R-EXP-03).
- Benchmarks: NExT-OE (test), MSVD-QA (test), ActivityNet-QA (test), all T1 (RESEARCH_DIRECTION §13).
- Sample sizes per benchmark: main 2,000 ⊃ annotated 300 ⊃ video-labelled subset 200 (with `vid_rank`, so any prefix of 150+ is a usable stratified sample). MSVD-QA annotated sample has ≥ 15 questions per type. Dev/test = 40/60 by question (R-ANN-05, R-ANN-16, RESEARCH_DIRECTION §13).

## Review Focus

1. **GPT-3.5 disappears early, or individual requests start returning `model_not_found` mid-run.** Expected: the runner stops cleanly, keeps every cached response, writes a dated retirement entry to `docs/decisions.md`, and the capture report shows exactly what was captured. Never retry forever or mark items as unjudged verdicts. *(Test in Task 12.)*
2. **The `pred` value arrives split across tokens, or its token is not in the top-20 logprobs.** Expected: `margin=None`, `parse_status="api_no_logprob"`, and the parsed verdict is still stored. Never a margin read off the wrong token. *(Test in Task 11.)*
3. **Missing or corrupt videos** (ActivityNet videos taken down from YouTube; truncated files). Expected: the item is excluded before sampling, with a reason in `docs/data_availability.md`, and samples are deterministic given the availability list. A decode failure during frame extraction raises with the video id; it never yields blank frames. *(Tests in Tasks 4 and 6.)*
4. **A generator returns an empty string, a refusal, or text with leading articles or punctuation only.** Expected: stored verbatim, and `normalize_answer` never maps a non-empty answer to `""` (it falls back to the lowercased stripped original). Refusals are kept as answers, not dropped. *(Test in Task 7.)*
5. **A question or answer containing `{`, `}`, quotes or newlines.** Expected: judge prompts render it literally, and the T-VCG parser still handles dict-like output with nested quotes. *(Test in Task 10.)*

---

### Task 1: Repository skeleton, config models, hashing, CI

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.github/workflows/ci.yml`
- Create: `src/vjaudit/__init__.py`, `src/vjaudit/hashing.py`, `src/vjaudit/config.py`, `src/vjaudit/cli.py`
- Create: `configs/benchmarks.yaml`, `configs/generators.yaml`, `configs/judges.yaml`, `configs/grids/stage1.yaml`, `configs/local/.gitkeep`
- Create: `docs/decisions.md`, `docs/superpowers/specs/` (copy `RESEARCH_DIRECTION.md` and `REQUIREMENTS_CONTRACT.md`), `docs/superpowers/plans/` (copy this plan)
- Test: `tests/test_hashing.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `canonical_json(obj) -> str`: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
  - `sha256_hex(s: str) -> str`
  - `config_hash(obj) -> str` = `sha256_hex(canonical_json(obj))`, 64 hex characters
  - `short_id(*parts: str) -> str` = `sha1("|".join(parts)).hexdigest()[:16]`
  - `git_sha() -> str`: `"nogit"` outside a repo
  - pydantic models `BenchmarkCfg`, `GeneratorCfg`, `JudgeCfg`, `GridCfg`, plus `load_yaml(path: Path, model: type[T]) -> T`
  - `GeneratorCfg` fields: `id`, `family`, `hf_repo`, `revision`, `backend: Literal["vllm","hf"]`, `dtype="bfloat16"`, `n_frames: dict[str,int]`, `max_model_len: int`, `styles: list[Literal["short","sentence"]]`, `blind: bool`, `frame_ablation: list[int]`, `quantization: str | None`
  - `JudgeCfg` fields: `id`, `kind: Literal["api","open"]`, `model`, `max_usd: float`, `price_in_per_mtok: float`, `price_out_per_mtok: float`, `concurrency: int`
- CLI: `python -m vjaudit.cli <command>` (Typer). Commands are added by later tasks.

- [ ] **Step 1: Write failing tests**

```python
def test_canonical_json_is_key_order_invariant():
    assert canonical_json({"b": 1, "a": [2, {"d": 3, "c": 4}]}) == canonical_json({"a": [2, {"c": 4, "d": 3}], "b": 1})

def test_short_id_matches_contract_formula():
    assert short_id("msvd:test:1", "qwen3vl8b", "abc") == hashlib.sha1(b"msvd:test:1|qwen3vl8b|abc").hexdigest()[:16]

def test_config_hash_changes_with_any_field():
    assert config_hash({"n_frames": 16}) != config_hash({"n_frames": 32})

def test_generators_yaml_validates(): ...   # load_yaml("configs/generators.yaml", GeneratorsFile) succeeds; ids unique
def test_local_dir_is_gitignored(): ...     # "configs/local/*.yaml" pattern present in .gitignore
```

- [ ] **Step 2: Run tests to confirm they fail:** `uv run pytest tests/test_hashing.py tests/test_config.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement** `hashing.py` and `config.py` with the interfaces above. Fill `configs/judges.yaml` with two judges:
  - `id: gpt-3.5-turbo-0125`, `kind: api`, `model: gpt-3.5-turbo-0125`, `max_usd: 100`, `price_in_per_mtok: 0.50`, `price_out_per_mtok: 1.50`, `concurrency: 16`;
  - `id: gpt-3.5-turbo-1106`, with the same fields except `model`, `max_usd: 20` and that snapshot's listed prices.

  Re-check prices on the OpenAI pricing page and record the date in a YAML comment. Fill `configs/generators.yaml` with the five defaults from Task 8; revisions start as `"UNPINNED"` and Task 8 pins them. Set `.github/workflows/ci.yml` to run `uv run pytest -m "not gpu and not api and not legacy"`.
- [ ] **Step 4: Run tests:** `uv run pytest tests/test_hashing.py tests/test_config.py -v`. Expected: all PASS. Then `make sanity` (alias for the CI command). Expected: PASS.
- [ ] **Step 5: Commit:** `git add -A && git commit -m "feat: repo skeleton, config models, hashing, CI"`

---

### Task 2: Record schemas and the content-addressed artifact store

**Files:**
- Create: `src/vjaudit/schemas.py`, `src/vjaudit/store.py`
- Test: `tests/test_schemas.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `canonical_json`, `config_hash`, `git_sha` (Task 1).
- Produces:
  - pydantic models `Item`, `Generation`, `JudgeCall`, with exactly the fields of contract §8.2, **plus these additions** (record them in `docs/decisions.md` as schema changes):
    - `Item.vid_rank: int | None`
    - `Item.available: bool`
    - `JudgeCall.sample_idx: int = 0`
    - `JudgeCall.decoding: dict`
    - `JudgeCall.api_model_returned: str | None`
    - `JudgeCall.system_fingerprint: str | None`
    - `JudgeCall.raw_text: str | None`
    - `JudgeCall.vcg_score: float | None`. The contract says int; the official prompt's own example emits 4.8.
  - `ArtifactStore(root: Path)` with:
    - `put(kind: str, record: BaseModel, key: str) -> bool`: appends and returns False if `key` exists
    - `has(kind, key) -> bool`
    - `iter(kind) -> Iterator[dict]`
    - `export_sorted(kind, out: Path) -> str`: writes JSONL sorted by key, excluding `created_at`, and returns its sha256
  - `stamp(record_dict: dict, cfg: dict) -> dict`: adds `config_hash`, `git_sha`, `created_at` (UTC ISO-8601)
  - Layout: `artifacts/<kind>/part-<pid>-<n>.jsonl`. Every line is `{"key": ..., "record": {...}}`. Each line is flushed and fsynced. Keys load into an in-memory set at open.

- [ ] **Step 1: Write failing tests**

```python
def test_put_is_idempotent(tmp_path):
    s = ArtifactStore(tmp_path); g = make_generation()
    assert s.put("generation", g, g.gen_id) is True
    assert s.put("generation", g, g.gen_id) is False
    assert len(list(s.iter("generation"))) == 1

def test_truncated_last_line_is_ignored_and_reported(tmp_path, caplog):
    ...  # write 3 records, then append half a JSON line; reopen; iter yields 3; a warning names the file

def test_interrupt_then_resume_equals_uninterrupted(tmp_path):
    ...  # run a fake job over 50 keys, raise after 20, rerun; export_sorted hash == hash from one clean run

def test_schema_rejects_unknown_parse_status():
    with pytest.raises(ValidationError): make_judgecall(parse_status="maybe")
```

- [ ] **Step 2: Run tests:** `uv run pytest tests/test_schemas.py tests/test_store.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `schemas.py` and `store.py` to the interfaces above.
- [ ] **Step 4: Run tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: record schemas and resumable content-addressed store"`

---

### Task 3: SP0 — prior-pipeline integrity check, regression of Tables 2–3, S1 label (Gate G0)

**Files:**
- Create: `configs/local/legacy.yaml` (gitignored). It lists one entry per cell: `{dataset, split, frames, condition, set, gen_path, judge_path}`, plus `prompts_dir`, `annotations_dir`, `inference_py: [paths]`.
- Create: `src/vjaudit/legacy/load.py`, `src/vjaudit/legacy/diff.py`, `src/vjaudit/legacy/tables.py`, `scripts/sp0_report.py`
- Test: `tests/legacy/test_diff.py`, `tests/legacy/test_tables.py`, `tests/legacy/expected_tables.json`, `tests/fixtures/legacy/` (tiny generation and judge JSONs in the prior format)

**Interfaces:**
- Legacy formats (from the prior repo README):
  - Generation JSON: `{"<idx>": {"Question", "Original Answer", "Generated Answer", "Similarity Score", "BERTScore"}}`
  - Judge JSON: the same, plus `"Similarity": "Yes" | "No"`
- Produces:
  - `load_legacy_cells(cfg_path) -> list[LegacyCell]`
  - `pairwise_identity(cells, split) -> pandas.DataFrame`, columns `[cond_a, set_a, cond_b, set_b, dataset, frames, n, pct_identical]`, comparing `"Generated Answer"` byte-for-byte on shared indices
  - `acceptance(judge_json) -> float` = 100 × Yes / total; unparseable values counted as No, matching the prior paper
  - `table2(cells, annotations_dir) -> dict`, `table3(cells) -> dict`

**`tests/legacy/expected_tables.json`** transcribes the camera-ready values.

Table 3: test-set acceptance %, 4 / 8 frames.

| Dataset | Cond | Frames | Zero | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|---|---|
| NExT | one | 4 | 55.11 | 71.20 | 58.74 | 58.40 | 53.06 | 63.56 |
| NExT | one | 8 | 54.72 | 71.67 | 58.57 | 58.89 | 52.31 | 63.47 |
| NExT | few | 4 | 55.11 | 57.86 | 57.39 | 82.22 | 55.24 | 53.62 |
| NExT | few | 8 | 54.72 | 58.12 | 57.36 | 82.92 | 55.37 | 54.33 |
| MSVD | one | 4 | 77.72 | 73.12 | 76.95 | 78.72 | 64.99 | 78.05 |
| MSVD | one | 8 | 76.73 | 72.36 | 76.01 | 77.54 | 64.12 | 77.04 |
| MSVD | few | 4 | 77.72 | 68.50 | 66.42 | 82.67 | 67.61 | 67.15 |
| MSVD | few | 8 | 76.73 | 68.15 | 66.03 | 83.36 | 66.01 | 67.44 |

Table 2: validation few-shot, J% / FP / FN; consensus C% = 56.5 for NExT and 63.5 for MSVD.

| Dataset | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| NExT | 58.0 / 33 / 30 | 86.0 / 66 / 7 | 55.5 / 40 / 42 | 55.0 / 33 / 36 |
| MSVD | 64.0 / 31 / 30 | 83.0 / 52 / 13 | 61.5 / 35 / 39 | 66.0 / 32 / 27 |

- [ ] **Step 1: Write failing tests.** `test_pairwise_identity_on_fixture` (two fixture cells that differ in 1 of 4 answers give 75.0) and `test_acceptance_counts_unparseable_as_no` run in CI. `@pytest.mark.legacy test_table3_reproduces_to_0p1` and `test_table2_reproduces` compare against `expected_tables.json` with `abs=0.1` (FP/FN exact).
- [ ] **Step 2: Run** `uv run pytest tests/legacy -v -m "not legacy"`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `load.py`, `diff.py`, `tables.py`. `scripts/sp0_report.py` writes `docs/sp0_report.md`, which contains:
  - the identity table for **val and test**;
  - Tables 2–3 reproduced next to the expected values;
  - S1's one-shot demonstration verdict label, read from `<prompts_dir>/examples/one_shot.json` (R-SP0-05);
  - the legacy quantization setting (the prior README states 4-bit).
- [ ] **Step 4: Run** `uv run pytest tests/legacy -v` on the machine that holds the legacy artifacts. Expected: PASS, including the `legacy` tests. If Table 2/3 cells miss by more than 0.1, list each miss in `docs/sp0_report.md` and do not proceed past Step 6 until explained.
- [ ] **Step 5: Investigate R-SP0-02 by hand.** Read each `inference_py` and determine whether the exemplar block is included in the **generator** prompt for val and test. Write a dated `docs/decisions.md` entry with exactly one outcome:
  - (a) by-design cache reuse;
  - (b) exemplars never reached the generator;
  - (c) other, with evidence (file:line).
- [ ] **Step 6: Gate G0.** If the outcome is (b): stop this task, notify all authors with the report, and record their decision on a BlackboxNLP correction (R-SP0-03). Tasks 4–14 do not depend on this outcome and continue in parallel. Only the paper's citation of prior results waits.
- [ ] **Step 7: Commit:** `git commit -am "feat(sp0): legacy diff + table regression + integrity decision"`. Commit the code and the report only; `configs/local/` stays untracked.

---

### Task 4: Benchmark loaders → `Item` records, with a video-availability check

**Files:**
- Create: `src/vjaudit/data/loaders/msvd.py`, `anetqa.py`, `nextqa.py`, `src/vjaudit/data/availability.py`
- Modify: `configs/benchmarks.yaml` (raw file names; video dir keys refer to `configs/local/paths.yaml`)
- Test: `tests/data/test_loaders.py`, with fixtures in `tests/fixtures/raw/{msvd,anetqa,nextqa}/` (5–10 rows each, in the documented formats below)

**Interfaces:**
- Produces: `load_benchmark(name: Literal["msvd","anetqa","nextoe"], raw_dir: Path, split="test") -> list[Item]`, `qtype_msvd(question: str) -> str`, `check_availability(items, video_root: Path, probe: Callable[[Path], bool] = probe_decodable) -> list[Item]` (sets `Item.available`, writes `docs/data_availability.md`).
- `item_id` = `"{benchmark}:{split}:{native_qid}"`.

**Raw formats.** Verify these against the real files on day 1. If a file differs, update the adapter **and** its fixture in the same commit.

| Benchmark | Source | Fields |
|---|---|---|
| MSVD-QA | `test_qa.json` | list of `{"id", "video_id", "question", "answer"}`; video file name via `youtube_mapping.txt` (`<youtube_clip> vid<id>`) |
| ActivityNet-QA | `test_q.json` + `test_a.json`, joined on `question_id` | q: `{"video_name", "question_id", "question"}`; a: `{"question_id", "answer", "type"}`; video file `v_<video_name>.*` |
| NExT-OE | `test.csv` (OE) joined with NExT-QA MC `test.csv` on `(video, qid)` | OE: `video, qid, question, answer, type`; MC adds `a0..a4` and `answer` (the index) → `mc_options`, `mc_answer_idx`; video path via `map_vid_vidorID.json`. If test-set answers are not available locally, use the split with answers that the prior paper used, and record it in `docs/decisions.md`. |

- **qtype rules:**
  - MSVD: first word of the lowercased question in {what, who, how, when, where}; otherwise `"other"`.
  - ActivityNet-QA: native `type` integer, mapped to its name.
  - NExT: native `type` code (CW, CH, TN, TC, TP, DL, DO, DC).

- [ ] **Step 1: Write failing tests.**
  - `test_msvd_item_ids_and_qtypes`: fixture row 3 gives `item_id == "msvd:test:3"` and `qtype == "where"`.
  - `test_anetqa_join_drops_unmatched_with_warning`
  - `test_nextoe_mc_join_attaches_options`: options length 5 and `mc_answer_idx` in range.
  - `test_duplicate_native_qid_raises`
  - `test_unavailable_video_is_flagged_not_dropped`: probe returns False for one video → `available=False`, and `docs/data_availability.md` lists it with reason `"missing"` or `"undecodable"`.
- [ ] **Step 2: Run** `uv run pytest tests/data/test_loaders.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the loaders. `probe_decodable(path)` opens the file with PyAV and decodes the first and last frame. Any exception counts as undecodable.
- [ ] **Step 4: Run tests.** Expected: PASS. Then on the server: `python -m vjaudit.cli data check-availability`. Expected: `docs/data_availability.md` written with per-benchmark counts. **If ActivityNet-QA has under 1,500 test videos available, tell the team the same day** (video acquisition becomes a blocker).
- [ ] **Step 5: Commit:** `git commit -am "feat(data): benchmark loaders and availability check"`

---

### Task 5: Nested stratified samples and the dev/test split

**Files:**
- Create: `src/vjaudit/data/sampling.py`
- Test: `tests/data/test_sampling.py`

**Interfaces:**
- Consumes: `Item` (Task 2), loaders (Task 4).
- Produces:
  - `allocate(counts: dict[str,int], n: int, min_per: int = 0) -> dict[str,int]`
  - `stratified_sample(items: list[Item], n: int, seed: int = 42, min_per: int = 0) -> list[Item]`
  - `assign_nested(items: list[Item], sizes=(2000, 300, 200), seed=42, msvd_min_per=15) -> list[Item]`: sets `in_main`, `in_annotated`, `in_vid_subset`, `vid_rank`
  - `split_dev_test(items: list[Item], frac_dev=0.4, seed=42) -> list[Item]`: sets `human_split` on annotated items only, by question, stratified by `qtype`
  - CLI `python -m vjaudit.cli data sample` writes `artifacts/item/` and `docs/samples.md`
- **Allocation algorithm** (pin it):
  1. Proportional allocation with the largest-remainder method.
  2. Raise any stratum below `min_per` (capped at its size) by taking one at a time from the stratum with the largest current allocation.
  3. Within each stratum, sort by `item_id`, then `random.Random(seed).sample`.
- **Nesting:** the annotated sample is drawn from the main sample, and the video subset from the annotated sample. `vid_rank` is a stratified ordering: round-robin across qtypes, each qtype in seeded order. Any prefix of length ≥ 150 is then approximately stratified.
- **Availability:** only items with `available=True` are eligible.

- [ ] **Step 1: Write failing tests.**
  - `test_allocate_sums_to_n_and_respects_min`
  - `test_nesting_holds`: every annotated item is in main; every vid item is annotated.
  - `test_msvd_min_15_per_type`
  - `test_dev_test_is_40_60_within_1pct_per_benchmark`
  - `test_same_seed_same_sample` (byte-identical export)
  - `test_unavailable_items_never_sampled`
  - `test_vid_rank_prefix_covers_all_qtypes`: the first 150 by `vid_rank` contain every qtype that has ≥ 3 vid items.
- [ ] **Step 2: Run** `uv run pytest tests/data/test_sampling.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `sampling.py`.
- [ ] **Step 4: Run tests.** Expected: PASS. On the server: `python -m vjaudit.cli data sample`. Expected: `docs/samples.md` shows 2000/300/200 per benchmark, the per-type counts, and the dev/test counts.
- [ ] **Step 5: Commit:** `git commit -am "feat(data): nested stratified samples and dev/test split"`

---

### Task 6: Frame extraction cache

**Files:**
- Create: `src/vjaudit/data/frames.py`
- Test: `tests/data/test_frames.py`. The test writes a tiny 40-frame video with PyAV in which frame `i` is a solid colour with red channel value `i*6`.

**Interfaces:**
- Produces:
  - `sample_indices(n_total: int, n: int, start: int = 0, end: int | None = None) -> list[int]` = `[start + int((i + 0.5) * L / n) for i in range(n)]`, where `L = (end or n_total) - start`. This is segment-centred and uniform.
  - `FrameCache(root: Path, max_side: int = 768)` with `.get(video_path: Path, video_id: str, n: int, t_start: float | None, t_end: float | None) -> tuple[np.ndarray, list[int]]`, returning a uint8 array of shape (n, H, W, 3) and the frame indices.
  - Cache key: `"{video_id}_{n}_centered_{t_start}_{t_end}.npz"`, storing frames + indices.
  - Judges later use the documented subset `indices[::2]` of the generator's frames (R-EXP-06). Note this in a docstring; nothing to implement now.

- [ ] **Step 1: Write failing tests.**
  - `test_sample_indices_formula`: `sample_indices(40, 4) == [5, 15, 25, 35]`.
  - `test_cache_returns_exact_frames`: red channel of returned frame k == `6 * indices[k]`.
  - `test_cache_hit_is_identical_and_skips_decode` (patch the decoder to raise on the second call).
  - `test_segment_restricts_range`
  - `test_corrupt_file_raises_with_video_id`
  - `test_resize_preserves_aspect_and_max_side`
- [ ] **Step 2: Run** `uv run pytest tests/data/test_frames.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with PyAV. Decode sequentially and keep only the requested indices. Use the stream's frame count, falling back to a full decode count when it is missing. Resize with PIL `Image.thumbnail((max_side, max_side))`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(data): uniform frame sampling and cache"`

---

### Task 7: Answer normalization

**Files:**
- Create: `src/vjaudit/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_answer(text: str) -> str`. Used for annotation units (SP4) and dedup reporting.
- **Rules, in order:**
  1. NFKC;
  2. lowercase;
  3. replace every `string.punctuation` character with a space;
  4. collapse whitespace and strip;
  5. strip **one** leading article from {a, an, the};
  6. if the result is empty but the input was not, return the lowercased, stripped input.

- [ ] **Step 1: Write failing test** `test_normalize_cases`, a 22-row `pytest.mark.parametrize` table that includes:
  - `("A man.", "man")`, `("the man", "man")`, `("Man", "man")` → all equal
  - `("an apple", "apple")`
  - `("The The", "the")`
  - `("playing guitar", "playing guitar")` ≠ `("playing a guitar", "playing a guitar")` (an inner article is kept)
  - `("man's hat", "man s hat")`
  - `("  Two\n dogs ", "two dogs")`
  - `("Ｄｏｇ", "dog")` (NFKC)
  - `("", "")`
  - `("...", "...")` (fallback; never empty)
  - `("a", "a")` (fallback)
  - `("Sorry, I can't tell from the video.", "sorry i can t tell from the video")`
  - `("yes", "yes")`, `("No.", "no")`
  - `("3", "3")`, `("three", "three")` (no number canonicalisation)
  - `("cat or dog", "cat or dog")`
  - `("a cat", "cat")` ≠ `("a cat or a dog", "cat or a dog")`
- [ ] **Step 2: Run** `uv run pytest tests/test_normalize.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `normalize_answer` per the rules.
- [ ] **Step 4: Run tests.** Expected: 22 PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: answer normalization with tricky-case table"`

---

### Task 8: Generator adapters, system expansion, vLLM engine

**Files:**
- Create: `configs/prompts/generation/short.txt`, `configs/prompts/generation/sentence.txt`, `configs/prompts/generation/mc.txt`
- Create: `src/vjaudit/generate/systems.py`, `src/vjaudit/generate/adapters.py`, `src/vjaudit/generate/engine.py`, `src/vjaudit/generate/runner.py`
- Modify: `configs/generators.yaml` (pin revisions)
- Test: `tests/generate/test_systems.py`, `tests/generate/test_adapters.py` (golden rendered prompts in `tests/generate/golden/<family>_<mode>.txt`), `tests/generate/test_runner.py`, `tests/generate/test_smoke_gpu.py` (marker `gpu`)

**Interfaces:**
- Consumes: `Item`, `Generation`, `ArtifactStore`, `FrameCache`, `normalize_answer`, `short_id`, `config_hash`.
- Produces:
  - `SystemSpec(system_id: str, generator_id: str, style: Literal["short","sentence","mc"], blind: bool, n_frames: int)`
  - `expand_systems(gens: list[GeneratorCfg], benchmark: str) -> list[SystemSpec]`: the 11 judged systems
  - `expand_mc_systems(gens: list[GeneratorCfg]) -> list[SystemSpec]`: `<gen>-mc` for the 5 core generators, used on NExT only (the G-ANCHOR anchor, which needs no judge)
  - `parse_mc_letter(text: str) -> int | None`: strip; if the first character is in A–E, use it; otherwise the first standalone `\b[A-E]\b`; map A..E to 0..4; otherwise None
  - `Adapter.build_request(item: Item, frames: np.ndarray | None, spec: SystemSpec) -> dict`, returning a vLLM input with keys `prompt` and `multi_modal_data`. Prompt text is rendered via the pinned HF processor's chat template.
  - `get_adapter(family: str) -> Adapter` for families `qwen-vl`, `internvl`, `llava-next-video`
  - `Engine` protocol: `generate(requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]` (text, n_output_tokens). Implementations: `VllmEngine(cfg)` and `FakeEngine(fn)`.
  - `run_generation(store, items, spec, engine, frame_cache, batch_size=16) -> int`: the number of new records
  - `gen_id = short_id(item_id, system_id, config_hash(gen_cfg_dict))`
- **Pinned prompt text:**
  - `short.txt` = `{question}\nAnswer the question using a single word or phrase.`
  - `sentence.txt` = `{question}\nAnswer in one complete sentence, then briefly say what in the video supports your answer.`
  - `mc.txt` = `{question}\nOptions:\nA. {a0}\nB. {a1}\nC. {a2}\nD. {a3}\nE. {a4}\nAnswer with the option's letter from the given choices directly.`, with `max_new_tokens` 8.
  - Use `str.replace`, never `format`, on every slot. This keeps braces in questions and options literal.
- **Default generators** (Step 3 pins the revision SHAs):

| id | HF repo | family | role |
|---|---|---|---|
| `qwen25vl7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | qwen-vl | blind=true |
| `qwen3vl8b` | `Qwen/Qwen3-VL-8B-Instruct` | qwen-vl | styles short+sentence; blind=true; frame_ablation [1, 4] |
| `qwen3vl2b` | `Qwen/Qwen3-VL-2B-Instruct` | qwen-vl | weak model |
| `internvl35_8b` | `OpenGVLab/InternVL3_5-8B` | internvl | blind=true |
| `llavanextvideo7b` | `llava-hf/LLaVA-NeXT-Video-7B-hf` | llava-next-video | continuity with the prior work |

  Fallbacks, if one fails the smoke test: `lmms-lab/LLaVA-Video-7B-Qwen2`, `DAMO-NLP-SG/VideoLLaMA3-7B` (via an `hf` backend adapter added only if needed). Record any substitution in `docs/decisions.md`.
- **System ids:** `<gen>` for the short style; `<gen>-sentence`; `<gen>-blind` (short style, no video); `<gen>-f<k>` for frame ablation. That is 11 systems in total.
- **Decoding:** temperature 0, `max_new_tokens` 128 (short) / 256 (sentence). Frames: 16 by default, 32 for ActivityNet-QA (`n_frames` map). Drop to 16 if the smoke test runs out of memory, and record it.

- [ ] **Step 1: Write failing tests.**
  - `test_expand_systems_yields_11_ids`
  - `test_blind_request_has_no_multimodal_data`
  - `test_question_with_braces_renders_literally` (`"What is {this}?"` appears verbatim in the prompt)
  - `test_golden_prompt_<family>_{video,blind}`: the rendered prompt equals the golden file
  - `test_runner_one_generation_per_item_system_and_resumes` (FakeEngine, interrupt after 7 of 20)
  - `test_empty_and_refusal_outputs_stored_verbatim`: FakeEngine returns `""` and `"Sorry, I cannot see the video."`; both records exist with that exact `output_text`.
  - `test_parse_mc_letter`: `"B"` → 1; `"(C) the man"` → 2; `"The answer is D."` → 3; `"none"` → None
  - `test_expand_mc_systems_yields_5`
  - `test_generation_record_fields`: `n_output_tokens`, `frame_indices`, `answer_norm == normalize_answer(output_text)`, `decoding == {"temperature": 0, "max_new_tokens": 128}`
- [ ] **Step 2: Run** `uv run pytest tests/generate -v -m "not gpu"`. Expected: FAIL.
- [ ] **Step 3: Implement** the adapters, the engine and the runner. To create the golden files, render once per family from the pinned processor, inspect each by eye (the video placeholder is present, the assistant turn is open), then freeze it. On the GPU server, resolve each repo's current revision SHA with `huggingface_hub.model_info(repo).sha` and write it into `generators.yaml`.
- [ ] **Step 4: Run** `uv run pytest tests/generate -v -m "not gpu"`. Expected: PASS. Then, on the server, `uv run pytest tests/generate/test_smoke_gpu.py -v -m gpu`. Each pinned generator answers 5 fixture items (video and blind) with non-empty text and no out-of-memory error at its configured frame count. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(generate): adapters, systems, vLLM engine, runner; pin generators"`

---

### Task 9: Throughput measurement and generation runs (9a: annotated subset; 9b: main sample)

**Files:**
- Create: `scripts/throughput.py`, `docs/throughput.md`
- Modify: `Makefile` (targets `generate-annotated`, `generate-main`), `configs/grids/stage1.yaml`
- Test: `tests/generate/test_coverage.py` (CPU, run on store exports)

**Interfaces:**
- Consumes: `run_generation`, `expand_systems`, the stores from Tasks 2 and 5.
- Produces:
  - `coverage(store, items, systems) -> dict[str, int]`, counting missing (item, system) pairs per system
  - `mc_accuracy(store, items, systems) -> dict[str, float]`: exact match of `parse_mc_letter(output_text)` against `mc_answer_idx`; unparseable counts as wrong and is reported separately
  - CLI:
    - `gen run --subset {annotated,main} [--mc]`
    - `gen coverage --subset {annotated,main}`
    - `gen rerun-check --n 50 --system <id>`
    - `gen mc-report` (writes `docs/mc_anchor.md`)
  - `make generate-annotated` (in_annotated items, 11 systems, 3 benchmarks ≈ 9,900 generations)
  - `make generate-main` (the remaining in_main items ≈ 56,100)
  - Both shard systems across the 3 GPUs with `CUDA_VISIBLE_DEVICES`, one process per GPU.

- [ ] **Step 1: Write failing test.** `test_coverage_reports_missing_pairs` on a fake store.
- [ ] **Step 2: Run it.** Expected: FAIL. Implement `coverage`. Run again. Expected: PASS.
- [ ] **Step 3: Measure throughput.** Run `python scripts/throughput.py --n 50` for every pinned generator at its configured frame count, plus blind. It writes items/s to `docs/throughput.md` and replaces the contract §10 estimates (R-SP0-07).
- [ ] **Step 4: 9a.** Run `make generate-annotated`. Then `python -m vjaudit.cli gen coverage --subset annotated`. Expected: `missing == 0` for all 11 systems × 3 benchmarks. **Target: Oct 11.**
- [ ] **Step 5: Reproducibility check.** Run `python -m vjaudit.cli gen rerun-check --n 50 --system qwen3vl8b`. It regenerates 50 items into a scratch store and compares `output_text`. Expected: 50/50 identical (SP2 acceptance). Any mismatch goes into `docs/decisions.md` with the vLLM version and the fix (for example, disabling prefix caching for generation).
- [ ] **Step 6: 9b.** Run `make generate-main`, then the coverage check on main. Expected: `missing == 0`. **Target: Oct 15** (otherwise use the fallback gate in the header).
- [ ] **Step 6b: MC anchor (not time-critical; run after 9b).** Run `python -m vjaudit.cli gen run --subset main --mc` for NExT, then `gen mc-report`. Expected: `docs/mc_anchor.md` gives exact-match accuracy and the unparseable rate for each of the 5 core generators on the same NExT main-sample questions.
- [ ] **Step 7: Commit:** `git commit -am "feat(generate): throughput doc, coverage tool, stage-1 generation runs"`. Commit `docs/throughput.md` and the manifest only; the artifacts stay untracked.

---

### Task 10: Judge templates T-VCG and T-RUB, and verdict parsers

**Files:**
- Create: `configs/prompts/judge/t_vcg_system.txt`, `configs/prompts/judge/t_vcg_user.txt`, `configs/prompts/judge/t_rub.j2`, `configs/prompts/judge/SOURCE.md`
- Create: `src/vjaudit/judge/templates.py`, `src/vjaudit/judge/parse.py`
- Test: `tests/judge/test_templates.py`, `tests/judge/test_parse.py`

**Interfaces:**
- Produces:
  - `Template(id: str, content_hash: str)`
  - `render_vcg(question, answer, pred) -> list[dict]` (system + user messages)
  - `render_rub(question, answer, pred) -> list[dict]`
  - `protocol_id(template: Template, decoding_tag: str) -> str` = `f"{template.id}@{template.content_hash[:8]}/{decoding_tag}"`
  - `parse_vcg(text: str) -> tuple[bool | None, float | None, str]` (verdict, score, status ∈ {"ok", "unparseable"})
  - `parse_yes_no(text: str) -> tuple[bool | None, str]` (status ∈ {"ok", "off_vocab"})
- **T-VCG = Video-ChatGPT `quantitative_evaluation/evaluate_activitynet_qa.py`, copied verbatim** (the system string and the user string with `{question}`, `{answer}`, `{pred}` slots). Steps:
  1. Record the upstream commit SHA in `SOURCE.md`.
  2. Diff the prompts in the MSVD/MSRVTT/TGIF scripts against it and record any difference in `docs/decisions.md`.
  3. Render by plain string replacement, never `str.format`.
  4. Reference text, for checking the copy only (the file is authoritative): system starts `"You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs."`; user ends `"For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."`
- **T-RUB:**
  1. Copy the prior repo's `src/Prompts/LLM_as_judge_base.j2` verbatim, recording its commit in `SOURCE.md`.
  2. Append the final line `Answer with a single word: Yes or No.` if the template does not already end with an equivalent instruction (R-ARC-01).
  3. Remove any exemplar slots; this is the zero-shot form.
- **`parse_vcg` algorithm:**
  1. Strip code fences.
  2. Take the first `{...}` span.
  3. Try `ast.literal_eval`.
  4. On failure, regex `pred['"]?\s*:\s*['"]?(yes|no)` and `score['"]?\s*:\s*([0-9.]+)`, case-insensitive.
  5. If pred is missing, return `(None, None, "unparseable")`.

- [ ] **Step 1: Write failing tests.**
  - `test_vcg_template_hash_is_frozen`: sha256 of the two files equals the constant committed with the copy.
  - `test_render_vcg_inserts_braces_quotes_newlines_literally`, using `pred="{'pred': 'no'}\nit's \"x\""`.
  - `test_parse_vcg_cases`:
    - `"{'pred': 'yes', 'score': 4}"` → `(True, 4.0, "ok")`
    - `'{"pred": "no", "score": 1}'` → `(False, 1.0, "ok")`
    - `"```python\n{'pred': 'yes', 'score': 4.8}\n```"` → `(True, 4.8, "ok")`
    - `"pred: yes score: 3"` → `(True, 3.0, "ok")`
    - `"I think so"` → `(None, None, "unparseable")`
  - `test_parse_yes_no`: `"Yes"`, `" yes."`, `"NO"` → ok; `"Maybe"` → `(None, "off_vocab")`
- [ ] **Step 2: Run** `uv run pytest tests/judge/test_templates.py tests/judge/test_parse.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**, copying the files as specified above.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(judge): T-VCG/T-RUB templates (copied verbatim) and parsers"`

---

### Task 11: Logprob margin readout for API responses

**Files:**
- Create: `src/vjaudit/judge/readout.py`
- Test: `tests/judge/test_readout.py`, fixtures `tests/fixtures/openai/*.json` (shaped like `choices[0].logprobs.content`: a list of `{token, logprob, top_logprobs:[{token, logprob}]}`)

**Interfaces:**
- Produces:
  - `token_class(tok: str) -> Literal["yes","no"] | None`: strip whitespace and the characters `'":`, lowercase, then match `yes`/`no` exactly
  - `margin_from_top(top: list[tuple[str, float]]) -> float | None` = `logsumexp(yes-class logprobs) − logsumexp(no-class logprobs)`; None if either class is absent
  - `p_yes(margin) -> float` = `1/(1+exp(-margin))`
  - `vcg_verdict_index(tokens: list[str]) -> int | None`
  - `readout_vcg(content) -> tuple[float | None, list[list]]` and `readout_first_token(content) -> tuple[float | None, list[list]]`. Each returns (margin, top-5 tokens at the verdict position).
- **`vcg_verdict_index` algorithm:**
  1. Concatenate tokens left to right, tracking the char offset.
  2. Find the first match of the regex `pred['"]?\s*:\s*['"]?` in the accumulated text.
  3. The verdict token is the first token starting at or after the match end whose `token_class` is not None.
  4. If the token covering that offset has no class (a split token such as `'y` + `es`), return None.

- [ ] **Step 1: Write failing tests.**
  - `test_margin_aggregates_variants`: top `[(" Yes",-0.1),("yes",-2.3),("No",-2.5)]` gives `logsumexp([-0.1,-2.3]) - (-2.5)` within 1e-9.
  - `test_margin_none_when_class_absent`
  - `test_vcg_index_on_standard_dict` (tokens `["{'", "pred", "':", " '", "yes", "',", ...]` → 4)
  - `test_vcg_index_double_quotes`
  - `test_vcg_index_split_token_returns_none`
  - `test_first_token_readout_for_rub`
- [ ] **Step 2: Run** `uv run pytest tests/judge/test_readout.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `readout.py`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(judge): logprob margin readout for T-VCG and first-token judges"`

---

### Task 12: OpenAI judge client (cache, retries, budget, retirement)

**Files:**
- Create: `src/vjaudit/judge/api_openai.py`
- Test: `tests/judge/test_api_openai.py` (an injected fake transport; no network)

**Interfaces:**
- Consumes: `canonical_json`, `sha256_hex`; `JudgeCfg`.
- Produces:
  - `OpenAIJudgeClient(cfg: JudgeCfg, cache_dir: Path, transport: Callable | None = None)`
  - `async complete(messages, *, max_tokens: int, temperature: float | None, logprobs: bool, top_logprobs: int, n: int = 1) -> dict`. Returns the raw response dict and adds `"_cached": bool`.
  - `request_hash(req: dict) -> str` = `sha256_hex(canonical_json(req))`. The request includes the model string.
  - `estimate_cost(reqs: list[dict]) -> float`: tiktoken `cl100k_base` input tokens, plus `max_tokens × n` output tokens, at the configured prices.
  - `spent_usd: float`, computed from the actual `usage` of non-cached responses.
  - Exceptions: `BudgetExceeded`, `ModelRetired`.
- **Behaviour:**
  - Cache: one JSON file per request hash under `cache_dir/<hash[:2]>/<hash>.json`, holding the request and the response.
  - Retries: on 429/5xx/timeouts, exponential backoff starting at 2 s, ×2, max 6 attempts, with jitter.
  - `model_not_found` or HTTP 404 on the model → raise `ModelRetired` immediately, no retry.
  - Before each non-cached call, if `spent_usd + estimate > max_usd` → raise `BudgetExceeded`.
  - An `asyncio.Semaphore(cfg.concurrency)` bounds concurrency.
  - Record `response["model"]` and `system_fingerprint`.

- [ ] **Step 1: Write failing tests.**
  - `test_cache_hit_makes_no_call`
  - `test_retries_429_then_succeeds` (the fake returns 429 twice, then 200; asserts 3 calls)
  - `test_model_not_found_raises_model_retired_without_retry`
  - `test_budget_exceeded_before_sending`
  - `test_request_hash_stable_across_dict_order`
  - `test_spent_usd_uses_usage_not_estimate`
- [ ] **Step 2: Run** `uv run pytest tests/judge/test_api_openai.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with `openai.AsyncOpenAI`. In tests, swap in the transport callable through the constructor.
- [ ] **Step 4: Run tests.** Expected: PASS. Then one live check, `uv run pytest -m api -k live_smoke`: a single real T-VCG call with `logprobs=True, top_logprobs=20` returns `choices[0].logprobs.content` and `model == "gpt-3.5-turbo-0125"`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(judge): cached, budgeted OpenAI judge client with retirement handling"`

---

### Task 13: Capture runner — JudgeCall records in priority order

**Files:**
- Create: `src/vjaudit/judge/capture.py`, `scripts/capture_report.py`, `docs/gpt35_capture.md` (generated)
- Modify: `configs/grids/stage1.yaml` (the capture stages), `Makefile` (target `capture-gpt35`)
- Test: `tests/judge/test_capture.py`

**Interfaces:**
- Consumes: `OpenAIJudgeClient`, `render_vcg`, `render_rub`, `protocol_id`, `parse_vcg`, `parse_yes_no`, `readout_vcg`, `readout_first_token`, `ArtifactStore`, `Generation`, `Item`.
- Produces:
  - `CaptureStage(name, judge_id: str, subset: Literal["annotated","main"], systems: Literal["core","ablation"], template: Literal["t-vcg","t-rub"], decoding_tag: str, request: dict)`. There is one `OpenAIJudgeClient` per `judge_id`.
  - `build_jobs(store, stages) -> list[Job]`: in stage order, then by `gen_id`; skips existing `judge_call_id`s
  - `async run_capture(store, client, jobs) -> CaptureSummary`
  - `judge_call_id = short_id(gen_id, judge_id, protocol_id, demo_set_id or "none", input_mode, str(sample_idx))`, with `input_mode="text"` and `demo_set_id=None`.
- **Pinned stages** (`core` = the 9 non-ablation systems; `main` excludes items already done in `annotated`):

| Stage | Subset | Systems | Template | decoding_tag | Request |
|---|---|---|---|---|---|
| S1 | annotated | core | t-vcg | `t0` | temperature 0, max_tokens 64, logprobs, top_logprobs 20 |
| S2 | annotated | core | t-rub | `t0` | temperature 0, max_tokens 1, logprobs, top_logprobs 20 |
| S3 | main | core | t-vcg | `t0` | as S1 |
| S4 | main | core | t-rub | `t0` | as S2 |
| S5 | annotated | core | t-vcg | `default-n3` | temperature omitted (the API default, as in the official script), n 3, logprobs, top_logprobs 20 → one JudgeCall per sample (`sample_idx` 0–2) |
| S6 | annotated, then main | ablation | t-vcg, t-rub | `t0` | as S1/S2 |
| S7 | annotated | core | t-vcg | `t0` | as S1, but judge `gpt-3.5-turbo-1106`. Measures snapshot-to-snapshot variance against humans. Skip with a decisions entry if the snapshot is unreachable. |

Stages S1–S6 use judge `gpt-3.5-turbo-0125`.

- On `ModelRetired`: stop, flush, and write a decisions entry `"gpt-3.5-turbo-0125 returned model_not_found at <UTC>; captured N calls"`.
- On `BudgetExceeded`: stop and report.
- **Before S1:** print `estimate_cost` for all stages and stop unless `--confirm` is given (R-BUD-01).

- [ ] **Step 1: Write failing tests.**
  - `test_jobs_follow_stage_priority_then_gen_id`
  - `test_resume_skips_existing_judge_calls`
  - `test_s5_writes_three_records_with_sample_idx`
  - `test_unparseable_is_recorded_with_status_not_dropped`
  - `test_judgecall_fields`: `judge_rev == "gpt-3.5-turbo-0125"`, `api_model_returned == response["model"]`, `margin`, `p_yes`, `top_tokens` length ≤ 5, `vcg_score`, `latency_s`, `cost_usd`
  - `test_no_logprob_keeps_parsed_verdict`: a fixture response with the `pred` token split across tokens gives `margin is None`, `parse_status == "api_no_logprob"` and `parsed_verdict is True`.
  - `test_s7_uses_1106_client`
  - `test_model_retired_stops_and_writes_decision`
- [ ] **Step 2: Run** `uv run pytest tests/judge/test_capture.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `capture.py` and `capture_report.py`. The report gives, per stage: calls done/expected, `parse_status` rates, the `api_no_logprob` rate, the yes-rate, cost, and the distinct `api_model_returned` / `system_fingerprint` values.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Live run.** `make capture-gpt35` (a dry run prints estimates), then `make capture-gpt35 CONFIRM=1`.
  - **Check after S1:** in `docs/gpt35_capture.md`, S1 parse `ok` ≥ 98% and `api_no_logprob` ≤ 5%. If not, stop and fix the parser or readout using the cached responses (no new calls needed), then continue.
  - **Targets:** S1–S2 by Oct 13; S3–S4 by Oct 19; S5–S7 by Oct 21.
- [ ] **Step 6: Commit:** `git commit -am "feat(judge): GPT-3.5 capture runner and report"`

---

### Task 14: Freeze and export the capture (by Oct 22)

**Files:**
- Create: `scripts/freeze_stage1.py`, `artifacts/manifest.jsonl` (tracked), `docs/stage1_freeze.md`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `ArtifactStore.export_sorted`, the API cache dir.
- Produces: `freeze(store, cache_dir, out_dir) -> list[ManifestEntry]` with `ManifestEntry(kind, path, n_records, sha256)`. It writes sorted exports for `item`, `generation` and `judgecall`, plus a tarball of the API cache with its sha256.

- [ ] **Step 1: Write failing test** `test_freeze_is_deterministic`: two freezes of the same store give an identical manifest.
- [ ] **Step 2: Run it.** Expected: FAIL. Implement. Run again. Expected: PASS.
- [ ] **Step 3:** Run `python scripts/freeze_stage1.py`. Then copy the tarball and exports to **two** storage locations (the server plus one off-server backup), and record the paths and checksums in `docs/stage1_freeze.md`.
- [ ] **Step 4:** Write a `docs/decisions.md` entry stating:
  - the final GPT-3.5 coverage per stage;
  - any fallback-gate use;
  - that no further GPT-3.5 calls are possible after 2026-10-23.
- [ ] **Step 5: Commit:** `git commit -am "chore: freeze stage-1 artifacts and GPT-3.5 cache"`

---

## Follow-on plans (not in this plan)

Each gets its own spec section and plan, in this order:

1. **SP4 annotation tooling.** Start now, in parallel. `L_ref`/`L_vid` UI with the new fields (annotator-written valid answers, the one-of-several reference label, E10, split E8), unit dedup on `answer_norm`, gold/qualification, agreement report, demonstration pool of 100 items/benchmark.
2. **SP3 open-weight and VLM judging engine.** Logit readout, matched text/frames pairs, CompassVerifier, VideoJudge, a frontier video judge, the reduced E8 sweep, plus judge throughput (the rest of R-SP0-07).
3. **SP8 literature survey.**
4. **SP6 analysis.** Error budget, HC mixed model, HC2 correction test, Δ_J − Δ_H, power-aware rankings, `final_eval.py`.
5. **SP7 protocol and legacy mapping.**
6. **SP5 natural refusal/hedge probes.**
7. **SP9 paper and release.**
