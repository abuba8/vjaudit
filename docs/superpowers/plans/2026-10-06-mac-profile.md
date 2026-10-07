# Mac Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole pipeline on a 32 GB Apple-Silicon Mac: small video models answer, small local judges grade, Claude labels by hand through export/import files, and a report compares judges with Claude's labels. The GPU-server path keeps working behind a profile switch.

**Architecture:**
- A *profile* (`configs/profiles/<name>/`) bundles generators, judges and the grid. `VJAUDIT_PROFILE` (or `--profile`) selects one, and every command loads it through `vjaudit.profile.load_profile`.
- Mac models run through Hugging Face `transformers` on MPS:
  - `TransformersEngine` generates answers;
  - `LocalTransport` judges. It returns OpenAI-shaped responses, so the existing client, capture, parsers, readout and report work unchanged.
- New packages: `vjaudit.annotate` (units, export, import) and `vjaudit.report` (metrics, summary).

**Tech Stack:** Python 3.11; torch ≥ 2.5 (MPS); transformers ≥ 4.57; accelerate; huggingface_hub; existing stack (pydantic, typer, PyAV, numpy, pandas, pytest).

**Spec:** `docs/superpowers/specs/2026-10-06-mac-profile-design.md` (approved 2026-10-06). The parent specs and the Stage-1 plan stay in force.

## Global Constraints

- **Device:** `mps` if available, else `cuda`, else `cpu`.
- **dtype:** `float16` on MPS, `bfloat16` on CUDA, `float32` on CPU.
- **Mac frames and samples:** `n_frames: {default: 8}`, `frame_max_side: 448`; `sample_sizes: {main: 1000, annotated: 100, vid: 30}`, `msvd_min_per: 15`, `frac_dev: 0.4`, `seed: 42`.
- **Mac generators** (all `backend: hf`):

| id | repo | family | styles | blind |
|---|---|---|---|---|
| `qwen25vl3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | qwen-vl | short, sentence | yes |
| `qwen3vl2b` | `Qwen/Qwen3-VL-2B-Instruct` | qwen-vl | short | — |
| `llavaov05b` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | llava-onevision | short | — |

- **Mac judges:**
  - `qwen3-4b` = `Qwen/Qwen3-4B-Instruct-2507`
  - `qwen25-1.5b` = `Qwen/Qwen2.5-1.5B-Instruct`
  - both: `kind: open`, `backend: transformers`, prices 0, `max_usd: 0`, `concurrency: 1`
  - plus the existing `gpt-3.5-turbo-0125` entry, used only by the optional stages.
- **Mac capture stages, in this order:** M1–M6 as in spec §4, then G1/G2 with `optional: true`. Optional stages run only when named in `--stages`.
- **Label sources:** must match `^(claude-manual(-[A-Za-z0-9_.]+)?|human:[A-Za-z0-9_.-]+)$`. Claude labels are never presented as human labels.
- **Export privacy:** exports never contain system ids, model names or judge verdicts (R-ANN-02).
- **Dependencies:** extra `mac` = `torch>=2.5`, `torchvision`, `transformers>=4.57`, `accelerate`, `huggingface_hub`. In the `gpu` extra, `vllm` gets `; sys_platform == 'linux'`.
- **Imports:** CPU tests never import torch. Torch is imported lazily inside default loaders only.
- **Stage-1 constraints still hold:** content-addressed resumable store, unparseable outputs counted and never mapped, `configs/local/` gitignored, `make sanity` green.

## Review Focus

1. **Claude's reply wraps the CSV in prose or a code fence, uses CRLF line endings, or quotes a note that contains commas.** Expected: import extracts the CSV block, parses it correctly, and rejects nothing that is valid. *(Test in Task 5.)*
2. **Claude skips units or invents unit ids.** Expected: invented ids are listed as errors and nothing is stored; skipped units are reported by `annotate status` as missing per batch. *(Test in Task 5.)*
3. **A local judge's tokenizer decodes Yes/No tokens with a leading space or as a separate piece** (`" Yes"`, `"Yes"`, `"yes"`). Expected: the margin still aggregates all variants. *(Test in Task 3.)*
4. **Memory pressure on the Mac when several models load in one run.** Expected: each generator's engine is released (`close()`) before the next generator loads. *(Test in Task 2.)*
5. **Running a Mac command without `PROFILE=mac`.** Expected: every command prints the active profile name, and records go to `artifacts/<profile>/`, so the profiles never mix. *(Test in Task 1.)*

---

### Task 1: Profiles

**Files:**
- Create: `src/vjaudit/profile.py`, `configs/profiles/mac/{generators,judges,grid}.yaml`
- Move (git mv, contents unchanged):
  - `configs/generators.yaml` → `configs/profiles/server/generators.yaml`
  - `configs/judges.yaml` → `configs/profiles/server/judges.yaml`
  - `configs/grids/stage1.yaml` → `configs/profiles/server/grid.yaml`
- Modify:
  - `src/vjaudit/config.py`: add `GridCfg.frame_max_side: int = 768`; add `"llava-onevision"` to `GeneratorCfg.family`; add `JudgeCfg.backend: Literal["openai","transformers"] = "openai"`, `JudgeCfg.revision: str = "UNPINNED"`, `JudgeCfg.chat_template_kwargs: dict = {}`.
  - `cli_data.py`, `cli_gen.py`, `cli_judge.py`, `scripts/throughput.py`, `scripts/freeze_stage1.py`: replace hard-coded config paths and `paths["artifacts"]` / `paths["api_cache"]` with the active profile's fields.
  - `generate/adapters.py` (`golden_render`): always reads `load_profile("server")`, because the golden prompts belong to the server models.
  - `cli_gen.py pin-revisions`: `--path` defaults to `<profile.root>/generators.yaml`.
- Test: `tests/test_profile.py`, plus a `tmp_paths` fixture in `tests/conftest.py` (a `paths.yaml` in `tmp_path` copying `configs/paths.example.yaml` with `artifacts: <tmp_path>/a`).
- Update these existing tests:
  - Path `configs/...` → `configs/profiles/server/...` in `tests/test_e2e.py`, `tests/test_config.py`, `tests/generate/{test_smoke_gpu,test_cli_gen,test_pin,conftest}.py`, `tests/judge/test_capture.py`.
  - `tests/data/test_cli_data.py`: `--grid <tmp grid>` becomes `--profile <tmp dir>` holding copies of the server YAMLs with the test grid as `grid.yaml`.
  - Tests that fill a store and then call a CLI command (`test_cli_gen.py`, `test_capture_cli.py`, `test_cli_data.py`): the store now lives at `<artifacts>/server`.

**Interfaces:**
- Produces:
  - `PROFILE_ENV = "VJAUDIT_PROFILE"`
  - `active_profile_name() -> str`: the env var or `"server"`
  - `Profile` (frozen dataclass): `name: str`, `root: Path`, `generators: list[GeneratorCfg]`, `judges: list[JudgeCfg]`, `grid: GridCfg`, `artifacts: Path`, `frame_cache: Path`, `api_cache: Path`, plus methods `store() -> ArtifactStore` and `judge(judge_id) -> JudgeCfg`
  - `load_profile(name_or_dir: str | Path | None = None, paths_file: Path = DEFAULT_PATHS_FILE) -> Profile`
    - an existing directory argument is used as the profile root (tests);
    - a name resolves to `CONFIGS / "profiles" / name`;
    - `artifacts = paths["artifacts"] / name`, `frame_cache = paths["frame_cache"] / name`, `api_cache = paths["api_cache"] / name`, where `name` is the root directory's name;
    - missing `frame_cache` / `api_cache` keys default to `<artifacts>/frames` and `<artifacts>/api_cache`.
  - Every CLI command takes `--profile TEXT` (default: env) and first prints `profile: <name>`.
  - The CLI options `--grid` and `--judges` are removed; `--paths` stays.
  - `make <target> PROFILE=mac` exports `VJAUDIT_PROFILE`.

- [ ] **Step 1: Write failing tests** in `tests/test_profile.py`:

```python
def test_server_profile_loads_moved_configs(tmp_paths):          # tmp paths.yaml with artifacts=tmp/a
    p = load_profile("server", tmp_paths)
    assert p.name == "server" and len(p.generators) == 5 and p.grid.sample_sizes["main"] == 2000
    assert p.artifacts == tmp_paths.parent / "a" / "server"

def test_mac_profile_values(tmp_paths):
    p = load_profile("mac", tmp_paths)
    assert [g.id for g in p.generators] == ["qwen25vl3b", "qwen3vl2b", "llavaov05b"]
    assert all(g.backend == "hf" and g.frames_for("anetqa") == 8 for g in p.generators)
    assert p.grid.frame_max_side == 448 and p.grid.sample_sizes == {"main": 1000, "annotated": 100, "vid": 30}
    j = p.judge("qwen3-4b")
    assert j.backend == "transformers" and j.model == "Qwen/Qwen3-4B-Instruct-2507" and j.max_usd == 0

def test_env_var_selects_profile(monkeypatch, tmp_paths):
    monkeypatch.setenv("VJAUDIT_PROFILE", "mac")
    assert load_profile(None, tmp_paths).name == "mac"

def test_profiles_write_to_separate_artifact_dirs(tmp_paths):
    assert load_profile("mac", tmp_paths).artifacts != load_profile("server", tmp_paths).artifacts

def test_profile_dir_argument(tmp_path, tmp_paths):              # copy configs/profiles/mac to tmp/p/mytest
    assert load_profile(tmp_path / "p" / "mytest", tmp_paths).name == "mytest"

def test_cli_prints_profile(tmp_paths):                          # `vjaudit.cli data sample --profile mac --paths ...` output starts with "profile: mac"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_profile.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `profile.py`. Write the mac YAMLs with the Global Constraints values and the stage table from spec §4 (`request` dicts as in the server grid; G1/G2 copy S1/S2 with `optional: true`). Move the server files and update every consumer.
- [ ] **Step 4: Run** `make sanity`. Expected: all PASS, including the updated older tests.
- [ ] **Step 5: Commit:** `git commit -am "feat: profiles (server, mac) with per-profile artifacts"`

---

### Task 2: Mac generation backend

**Files:**
- Create: `src/vjaudit/device.py`, `src/vjaudit/generate/hf_engine.py`, `src/vjaudit/generate/factory.py`, `src/vjaudit/generate/smoke.py`
- Modify:
  - `generate/adapters.py`: add `LlavaOneVisionAdapter` (family `llava-onevision`, same messages as qwen-vl).
  - `generate/coverage.py`: `run_subset` takes `renderer_factory: Callable[[GeneratorCfg, Engine], Renderer]` and calls `engine.close()` when the engine has it, after the generator's work.
  - `cli_gen.py`: `gen run` uses `make_engine` / `make_renderer`; the frame cache uses `profile.grid.frame_max_side`; new `gen smoke`.
- Test: `tests/generate/test_hf_engine.py`, `tests/generate/test_factory.py`; update the `renderer_factory` lambdas in existing tests to `lambda g, e: FakeRenderer()`.

**Interfaces:**
- `device.py`:
  - `pick_device(torch_mod=None) -> str` (mps → cuda → cpu; the injectable module is for tests)
  - `default_dtype(device: str) -> str` ("float16" / "bfloat16" / "float32")
- `hf_engine.py`:
  - `TransformersEngine(gen: GeneratorCfg, device: str | None = None, allow_unpinned: bool = False, loader: Callable | None = None)`
    - `loader(repo, revision, dtype, device) -> (processor, model)`; the default loader uses `AutoProcessor` + `AutoModelForImageTextToText`.
  - Methods: `render(messages) -> str`; `generate(requests: list[dict], max_new_tokens: int) -> list[tuple[str, int]]`; `close() -> None` (drops the model and frees MPS/CUDA cache when torch is present).
  - `generate`, per request:
    1. `inputs = processor(text=[prompt], videos=[frames] if "multi_modal_data" in req else None, return_tensors="pt").to(device)`;
    2. `out = model.generate(**inputs, do_sample=False, max_new_tokens=n)`;
    3. `new = out[0][inputs["input_ids"].shape[1]:]`;
    4. `text = processor.batch_decode([new], skip_special_tokens=True)[0].strip()`;
    5. return `(text, len(new))`.
  - An exception whose message contains "out of memory" is re-raised as `RuntimeError` with the hint "reduce n_frames or frame_max_side in configs/profiles/mac".
  - Any loader exception is re-raised as `RuntimeError("could not load <hf_repo>@<revision>: <original error>")`.
  - `UNPINNED` is refused unless `allow_unpinned` (same message style as `VllmEngine`). Normally the Mac user runs `gen pin-revisions` first (Task 3 extends it to judges), so the flag is only for quick tests.
- `factory.py`:
  - `make_engine(gen: GeneratorCfg, allow_unpinned: bool = False) -> Engine` (`vllm` → `VllmEngine`, `hf` → `TransformersEngine`)
  - `make_renderer(gen: GeneratorCfg, engine) -> Renderer` (`TransformersEngine` → `engine.render`; otherwise `HFChatRenderer`)
- `smoke.py`: `smoke_generators(profile, items: list[Item], frame_cache, engine_factory=make_engine, renderer_factory=make_renderer, n_items=2) -> list[dict]`
  - one row per (generator, item, mode ∈ {video, blind}) with keys `generator, item_id, mode, answer, seconds, device`;
  - `gen smoke` prints these rows as a table, using the first 2 available annotated items (error "run `make data-sample` first" if there are none).
- `gen run` / `gen smoke` accept `--allow-unpinned` (Mac convenience; prints a warning that results are not paper-grade).

- [ ] **Step 1: Write failing tests**:

```python
def test_pick_device_order():                       # fake torch modules: mps→"mps"; cuda only→"cuda"; none→"cpu"
def test_default_dtype():                           # {"mps":"float16","cuda":"bfloat16","cpu":"float32"}
def test_engine_video_and_blind_calls(fake_loader): # FakeProcessor records kwargs: videos=[frames] for video, videos=None for blind;
                                                    # returns ("a dog", 2) when model.generate appends ids [7, 8] decoded to " a dog \n"
def test_engine_refuses_unpinned():                 # ValueError mentioning "UNPINNED"
def test_engine_oom_hint():                          # model.generate raises RuntimeError("MPS backend out of memory") → message contains "reduce n_frames"
def test_engine_load_error_names_repo():            # loader raises OSError("404") → RuntimeError containing "Qwen/Qwen2.5-VL-3B-Instruct" and "404"
def test_make_engine_selects_backend():             # backend hf → TransformersEngine (fake loader); vllm → VllmEngine is not constructed in CPU tests (monkeypatched class)
def test_run_subset_closes_engine_before_next_generator(tmp_path):  # fake engines record close(); order: gen0.close before gen1 created
def test_llava_onevision_adapter_video_and_blind(): # same structure as qwen-vl adapter tests
def test_smoke_generators_rows(tmp_path):           # 1 fake generator, 2 items → 4 rows, keys exactly as above
```

- [ ] **Step 2: Run** `uv run pytest tests/generate -q -m "not gpu and not hf"`. Expected: FAIL.
- [ ] **Step 3: Implement** the files above. CPU code paths must not import torch (`close()` and the default loader import it lazily inside `try`).
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(generate): transformers/MPS engine, factory, llava-onevision adapter, gen smoke"`

---

### Task 3: Local judge transport

**Files:**
- Create: `src/vjaudit/judge/local.py`, `src/vjaudit/judge/factory.py`, `src/vjaudit/judge/smoke.py`
- Modify:
  - `cli_judge.py`: new `judge smoke [--allow-unpinned]`.
  - `generate/pin.py`: `pin_revisions` also accepts `model:` as the repo key, and pins only entries that have a `revision:` line (OpenAI judges have none, so they are untouched).
  - `cli_gen.py pin-revisions`: pins the profile's `generators.yaml` and then its `judges.yaml`.
- Test: `tests/judge/test_local.py`, `tests/generate/test_pin.py` (new case)

**Interfaces:**
- `content_from_logits(gen_ids: list[int], step_logits: list[np.ndarray], decode: Callable[[int], str], k: int) -> list[dict]`. For each step `i`:
  - `lp = log_softmax(step_logits[i])` (numerically stable, float64);
  - the entry is `{"token": decode(gen_ids[i]), "logprob": lp[gen_ids[i]], "top_logprobs": [{"token": decode(j), "logprob": lp[j]} for the k largest j, descending]}`.
- `JudgeBackend` protocol:
  - `render(messages: list[dict]) -> str`
  - `generate_once(prompt: str, max_new_tokens: int, temperature: float, seed: int) -> tuple[list[int], list[np.ndarray], int]` (generated ids, raw logits per step, prompt tokens); `temperature == 0` means greedy
  - `decode(token_id: int) -> str` (one token, for the logprob entries)
  - `decode_all(ids: list[int]) -> str` (the full answer text, special tokens skipped)
- `HFCausalBackend(cfg: JudgeCfg, device: str | None = None, allow_unpinned: bool = False)`:
  - default `JudgeBackend`; the model loads lazily on first use;
  - uses `AutoTokenizer` + `AutoModelForCausalLM` at `cfg.revision`, and refuses `UNPINNED` unless `allow_unpinned`;
  - `render` applies `apply_chat_template(..., add_generation_prompt=True, tokenize=False, **cfg.chat_template_kwargs)`;
  - `generate_once` calls `generate(output_logits=True, return_dict_in_generate=True)`, with `torch.manual_seed(seed)` when sampling.
- `LocalTransport(cfg: JudgeCfg, backend: JudgeBackend | None = None, allow_unpinned: bool = False)`:
  - `async __call__(req: dict) -> dict` runs `_run` in `asyncio.to_thread`;
  - `_run` renders `req["messages"]`, then for `k in range(req.get("n", 1))` calls `generate_once` with `temperature = 0.0` if `req.get("temperature") == 0`, else `req.get("temperature") or 1.0`, and `seed = k`;
  - it returns the OpenAI-shaped dict from spec §4: `choices[k].message.content = decode_all(ids)`, `choices[k].logprobs.content = content_from_logits(..., k=req["top_logprobs"])`, `system_fingerprint = cfg.revision`;
  - any exception → `TransportError(422, "local_error", str(e))`.
- `factory.make_client(cfg: JudgeCfg, cache_dir: Path, backend: JudgeBackend | None = None, allow_unpinned: bool = False) -> OpenAIJudgeClient` (transformers → `OpenAIJudgeClient(cfg, cache_dir, transport=LocalTransport(...))`; openai → default transport).
- `smoke.smoke_judges(profile, client_factory=make_client) -> list[dict]`:
  - runs one T-VCG and one T-RUB call per judge with `backend: transformers`, on `("what is the man doing?", "cooking", "he is cooking food")`;
  - uses a fresh temporary cache directory, so every smoke run really calls the model;
  - reads the result with the existing `parse_vcg` / `readout_vcg` (T-VCG) and `parse_yes_no` / `readout_first_token` (T-RUB);
  - rows have keys `judge, template, verdict, margin, top_tokens, seconds`.

- [ ] **Step 1: Write failing tests**:

```python
def test_content_from_logits_values():
    logits = [np.array([0.0, 1.0, 2.0]), np.array([3.0, 0.0, 0.0])]
    out = content_from_logits([2, 0], logits, decode=lambda i: [" No", " Yes", "Yes"][i], k=2)
    lse0 = math.log(math.exp(0) + math.exp(1) + math.exp(2))
    assert out[0]["token"] == "Yes" and out[0]["logprob"] == pytest.approx(2 - lse0)
    assert [t["token"] for t in out[0]["top_logprobs"]] == ["Yes", " Yes"]
    assert out[1]["top_logprobs"][0]["token"] == " No"

def test_margin_from_local_content_aggregates_space_variants():   # readout_first_token(content) with " Yes"/"Yes" summed (Review Focus 3)

async def test_local_transport_shapes_response(fake_backend):     # FakeBackend: render joins contents; generate_once returns ids/logits
    r = await LocalTransport(CFG, fake_backend)({"model": "m", "messages": MSGS, "max_tokens": 1, "n": 1,
                                                 "logprobs": True, "top_logprobs": 2, "temperature": 0})
    assert r["choices"][0]["message"]["content"] == "Yes" and r["usage"]["completion_tokens"] == 1
    assert len(r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]) == 2

async def test_local_transport_n3_uses_sampling_seeds(fake_backend):   # temperature absent → 3 calls with temperature 1.0, seeds 0,1,2
async def test_local_transport_error_is_422(fake_backend_raising):     # TransportError status 422, code "local_error"
async def test_client_with_local_transport_caches(tmp_path, fake_backend):  # make_client(cfg, tmp, backend) twice → backend called once
def test_smoke_judges_rows(tmp_path):                                   # 1 local judge → 2 rows with the keys above
def test_hf_backend_refuses_unpinned():                                 # HFCausalBackend(cfg revision UNPINNED) raises ValueError on first use
# tests/generate/test_pin.py
def test_pin_judges_only_entries_with_revision(tmp_path):              # mac judges.yaml copy: qwen3-4b, qwen25-1.5b pinned; gpt-3.5 block unchanged
```

- [ ] **Step 2: Run** `uv run pytest tests/judge/test_local.py tests/generate/test_pin.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(judge): local transformers judge transport with OpenAI-shaped logprobs; judge smoke"`

---

### Task 4: Capture with local judges and optional stages

**Files:**
- Modify:
  - `judge/capture.py`: `CaptureStage.optional: bool = False`; `select_stages(stages, names: list[str]) -> list[CaptureStage]` returns all non-optional stages if `names` is empty, else exactly the named ones in config order.
  - `cli_judge.py`: `judge capture` uses the profile, `select_stages` and `make_client(cfg, profile.api_cache, allow_unpinned=allow_unpinned)`; new option `--allow-unpinned`; the dry-run estimate shows `0.00` for local judges plus the job count.
  - `cli_judge._groups`: with no ablation systems, `core` = all judged systems.
- Test: `tests/judge/test_capture_local.py`

- [ ] **Step 1: Write failing tests**:

```python
def test_mac_stage_order_and_optional():          # names == ["M1",…,"M6","G1","G2"]; G1/G2 optional; select_stages(all, []) excludes G1,G2
def test_select_named_optional_stage():           # select_stages(all, ["G1"]) == [G1]
async def test_capture_with_local_judge_end_to_end(tmp_path):
    # store with 2 items × 2 systems; clients = {"qwen3-4b": make_client(cfg, tmp, FakeBackend yes-biased)}
    # run M1,M2 → 8 JudgeCall records, all parse_status "ok", margins not None, judge_rev == cfg.revision
def test_groups_without_ablation_is_all_systems():  # mac profile → core has 5 ids, ablation empty
```

- [ ] **Step 2: Run** `uv run pytest tests/judge/test_capture_local.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(judge): mac capture stages, optional stages, local clients in capture"`

---

### Task 5: Annotation units, Claude/human export, import

**Files:**
- Create: `src/vjaudit/annotate/{__init__,units,export,importer}.py`, `src/vjaudit/cli_annotate.py` (group `annotate`)
- Modify: `schemas.py` (add `Unit`, `Label` records), `cli.py`
- Test: `tests/annotate/test_units.py`, `test_export.py`, `test_import.py`

**Interfaces:**
- Records:
  - `Unit(Record)`: `unit_id, item_id, benchmark, answer_norm, display_answer, gen_ids: list[str], system_ids: list[str]`
  - `Label(Record)`: `label_id, unit_id, source, task: Literal["ref","vid"], label: Literal["correct","incorrect"], error_tags: list[str], note: str, batch: str`
- `build_units(items: list[Item], generations: list[dict]) -> list[Unit]`:
  - one unit per `(item_id, answer_norm)`, considering only generations whose item is in `items`;
  - `unit_id = short_id(item_id, answer_norm)`;
  - `display_answer` = the `output_text` of the generation with the smallest `gen_id`;
  - output sorted by `(item_id, unit_id)`.
- `plan_batches(units, batch_size=100) -> list[list[Unit]]`: units are grouped by `item_id` in order, and a question is never split. A batch closes when adding the next question would exceed `batch_size`, unless the batch is empty.
- `export_claude(units, items, out_dir) -> list[Path]` writes `batch_NN.md` (NN from 01) using the exact template below, plus `manifest.json` (`{"batch_01": [unit_ids…], …}`).
- `export_human(units, items, out_dir) -> list[Path]` writes `batch_NN.csv` with header `unit_id,benchmark,question,reference_answers,candidate_answer,label,error_tags,note`; reference answers are joined with `" | "`; it also writes the same manifest.
- `parse_label_csv(text: str) -> list[dict]` takes the content of the first fenced code block if one exists, otherwise the whole text. It finds the header line starting with `unit_id`, parses it with `csv.DictReader`, and trims whitespace. `error_tags` is split on `;`, with empties dropped.
- `import_labels(store, rows, source, manifest: dict, task="ref") -> ImportResult(stored, skipped_existing, missing_by_batch: dict[str, int])`:
  - validates everything first and raises `LabelImportError(errors: list[str])` with `"row <n>: <problem>"` lines;
  - stores only when there are no errors;
  - `label_id = short_id(unit_id, source)`.
- `validate_source(source) -> None` applies the Global Constraints regex.
- CLI:
  - `annotate export --set {annotated,main} --for {claude,human} [--batch-size 100]` writes `annotation/<profile>/<for>/` and stores the units;
  - `annotate import --source S --file F [--file …]`. The manifest is read from `annotation/<profile>/claude/` for `claude-manual*` sources and from `annotation/<profile>/human/` for `human:*` sources. Each label's `batch` is the manifest batch that holds its unit;
  - `annotate status` prints a table of batch / units / labelled per source / missing.
- **Exact `batch_NN.md` template** (`{…}` filled by code; code fences shown literally):

````text
# VideoQA answer labelling — batch {NN} of {TOTAL}

You are labelling answers to questions about short videos. You will NOT see the videos. For each
candidate answer, decide whether it is correct **given the question and the reference answer(s)**.

## Rules
1. Correct if it means the same as any reference answer, including synonyms and paraphrases.
2. More specific is fine ("puppy" for "dog"); more general is wrong ("animal" for "dog") unless the
   question asks for a category.
3. Partial answers to a multi-part reference are incorrect unless the missing part is not asked for.
4. Listing several alternatives ("a cat or a dog") is incorrect. Hedging around one answer: judge the answer.
5. Refusals, "cannot tell", and non-answers are incorrect.
6. Ignore extra details you cannot check, unless they contradict the reference.

## Error tags (for incorrect answers; optional for correct ones)
E1 synonym/paraphrase · E2 too general/too specific · E3 partial · E4 extra wrong detail ·
E5 shares words with the reference but means something else · E6 several alternatives ·
E7 refusal/non-answer · E8 reference looks wrong or ambiguous · E9 question ambiguous

## Output format
Reply with ONE csv code block and nothing else. Header exactly:
`unit_id,label,error_tags,note`
One row per unit below. label is `correct` or `incorrect`. error_tags separated by `;` (may be empty).
note is optional; quote it if it contains a comma.

## Items
### Q{k}. {question}
Reference answer(s): {ref1} | {ref2}
| unit_id | candidate answer |
|---|---|
| {unit_id} | {display_answer} |
````

- [ ] **Step 1: Write failing tests**:

```python
def test_units_dedup_by_normalized_answer():   # gens "A dog."/"dog"/"cat" on one item from 3 systems → 2 units; system_ids of the dog unit == both systems
def test_batches_never_split_a_question():     # 3 questions with 60/50/10 units, batch_size 100 → [[q1], [q2, q3]]
def test_claude_export_contains_rules_and_no_system_ids(tmp_path):
    # md contains "unit_id,label,error_tags,note" and "### Q1."; no system id, repo name or "qwen" in any file
def test_human_export_header(tmp_path):        # first line == "unit_id,benchmark,question,reference_answers,candidate_answer,label,error_tags,note"
def test_parse_fenced_crlf_quoted_reply():     # "Sure!\r\n```csv\r\nunit_id,label,error_tags,note\r\nu1,correct,,\"ok, fine\"\r\n```\r\nDone"
                                               # → [{"unit_id": "u1", "label": "correct", "error_tags": [], "note": "ok, fine"}]
def test_import_rejects_invented_ids_and_bad_labels_atomically(tmp_path):
    # rows with unknown id "zzz", label "maybe", tag "E99" → LabelImportError listing 3 errors; store has 0 labels
def test_import_reports_missing_units_per_batch(tmp_path):   # 4 units in batch_01, 3 labelled → missing_by_batch == {"batch_01": 1}
def test_reimport_skips_existing(tmp_path)                    # same file twice → stored 3 then skipped_existing 3
@pytest.mark.parametrize("src,ok", [("claude-manual", True), ("claude-manual-v2", True), ("human:ann1", True),
                                    ("human", False), ("claude", False), ("gpt", False)])
def test_validate_source(src, ok)
```

- [ ] **Step 2: Run** `uv run pytest tests/annotate -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(annotate): units, Claude/human export, validated label import"`

---

### Task 6: Report

**Files:**
- Create: `src/vjaudit/report/{__init__,metrics,summary}.py`, `src/vjaudit/cli_report.py` (group `report`)
- Modify: `cli.py`
- Test: `tests/report/test_metrics.py`, `tests/report/test_summary.py`

**Interfaces:**
- `metrics.py`:
  - `cohen_kappa(a: list[bool], b: list[bool]) -> float | None` (None when expected agreement is 1 or the lists are empty)
  - `error_rates(judge: list[bool], ref: list[bool]) -> dict` with keys `fpr, fnr, n_ref_pos, n_ref_neg`; a rate is None when its denominator is 0
  - `yes_rate(xs: list[bool]) -> float | None`
- `summary.py`:
  - `judge_configs(judgecalls) -> list[str]`: sorted keys `"<judge_id>/<template>/<decoding_tag>"`, built from the `judge_id` field and the template and decoding parts of `protocol_id` (`template@hash8/decoding_tag`); only `sample_idx == 0` and `input_mode == "text"` rows are used.
  - `build_tables(store, items, subset: str) -> dict`. Per benchmark and system:
    - `n`;
    - per config: `acc` (yes-rate of `parsed_verdict` over rows where it is not None) and `unparseable` (rows where it is None);
    - per label source: `acc`;
    - per (config, first source): `gap_pts` (100 × (acc_judge − acc_ref) on the generations that have both a verdict and a label), `kappa`, `fpr`, `fnr`.
    - Generations map to labels via `unit_id = short_id(item_id, answer_norm)`.
  - `visual_gain(tables) -> dict`: for every system `S` with a `S-blind` partner, `Δ = acc(S) − acc(S-blind)` per config and per source.
  - `write_summary(store, items, profile_name, out: Path) -> None`. It writes:
    1. the run-1 tables (annotated);
    2. the visual-gain lines;
    3. the run-2 table (main; configs only).
    Missing values show as "—". A header note states: "Label sources other than `human:*` are reference judges, not human ground truth."
- CLI: `report summary [--out docs/results_<profile>.md]`

- [ ] **Step 1: Write failing tests**:

```python
def test_kappa_and_rates_hand_values():
    j, r = [True, True, False, True], [True, False, False, True]
    assert cohen_kappa(j, r) == pytest.approx(0.5)
    assert error_rates(j, r) == {"fpr": 0.5, "fnr": 0.0, "n_ref_pos": 2, "n_ref_neg": 2}
def test_kappa_undefined_when_constant():        # cohen_kappa([True]*3, [True]*3) is None
def test_summary_tables_and_gap(tmp_path):       # 4 generations of one system, verdicts as j, claude labels as r → gap_pts == 25.0, kappa 0.5
def test_visual_gain(tmp_path):                  # S acc 0.75, S-blind acc 0.25 under one config → Δ == 0.5
def test_summary_without_labels_still_writes(tmp_path)  # no Label records → file written, "—" in source columns, header note present
```

- [ ] **Step 2: Run** `uv run pytest tests/report -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat(report): judge-vs-reference summary with kappa, error rates, visual gain"`

---

### Task 7: Dependencies and Makefile

**Files:**
- Modify:
  - `pyproject.toml`: add the `mac` extra; the `vllm` marker; pytest marker `mac`; `[tool.uv] conflicts = [[{extra = "gpu"}, {extra = "mac"}]]` if `uv lock` cannot resolve both together.
  - `uv.lock`
  - `Makefile`:
    - `PROFILE ?= server` and `export VJAUDIT_PROFILE = $(PROFILE)`;
    - `CI_MARKERS` gains `and not mac`;
    - `UNPINNED_FLAG = $(if $(ALLOW_UNPINNED),--allow-unpinned,)`, appended to every `gen run`, `gen smoke`, `judge smoke` and `judge capture` call.
  - New targets, each a plain CLI call (single process, no shards):

| Target | Command |
|---|---|
| `pin` | `gen pin-revisions` (generators and judges of the profile) |
| `smoke` | `gen smoke` then `judge smoke` |
| `run1` | `gen run --subset annotated` then `gen coverage --subset annotated` |
| `run2` | `gen run --subset main` then `gen coverage --subset main` |
| `judge-run1` | `judge capture --stages M1,M2,M3,M4 --confirm` |
| `judge-run2` | `judge capture --stages M5,M6 --confirm` |
| `annotate-export` | `annotate export --set annotated --for claude` |
| `report` | `report summary` |

- Create: `tests/test_mac_real.py`, marked `mac` (the spec §11 real-model check; never run by `make sanity`):
  - `test_real_generator_answers` — `smoke_generators` on `qwen3vl2b` with 1 item returns non-empty answers for video and blind;
  - `test_real_judge_reads_margin` — `smoke_judges` on `qwen25-1.5b` returns a non-None margin for T-RUB.
- Test: `tests/test_tooling.py`

- [ ] **Step 1: Write failing tests**:
  - `test_makefile_has_mac_targets`: each target name appears at the start of a line, followed by `:`;
  - `test_ci_markers_exclude_mac`: the `CI_MARKERS` line contains `not mac`;
  - `test_pyproject_mac_extra`: (tomllib) the `mac` extra contains `transformers>=4.57`, the gpu `vllm` requirement contains `sys_platform == 'linux'`, and the pytest markers include `mac`.
- [ ] **Step 2: Run** `uv run pytest tests/test_tooling.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Run `uv lock`; it must succeed.
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "build: mac extra, linux-only vllm, mac Makefile targets"`

---

### Task 8: Documentation

**Files:**
- Create: `docs/EXPLAINED.md`
- Modify: `docs/GUIDE.md` (profile section, updated config paths, link to EXPLAINED.md), `README.md` (link)
- Test: `tests/test_docs.py`

**`docs/EXPLAINED.md` outline** (plain language, short sentences, examples):

1. The idea in one page: what a judge is; the three ways a score goes wrong; what the project tests.
2. A worked example: one question, three answers, two judges, one reference label, traced end to end.
3. The pipeline in 8 steps (data → samples → frames → answers → judging → labels → report → freeze), each with "what / why / where in the code".
4. The code tour: one short paragraph per package and its key functions.
5. The Mac runbook, with exact commands, expected outputs and time estimates:
   1. install;
   2. paths;
   3. data check;
   4. sample;
   5. pin model revisions (`make pin PROFILE=mac`);
   6. smoke;
   7. verify prompts (or `--allow-unverified`);
   8. run 1;
   9. judge run 1;
   10. Claude export, labelling and import;
   11. report;
   12. run 2 + judge run 2;
   13. report.
6. Labelling with Claude, step by step, including what to do when Claude's reply is cut off or wrong.
7. How to read `docs/results_mac.md`.
8. What the Mac run can and cannot claim (no human labels yet; small models; T-RUB reconstruction unless vendored).
9. Glossary.

- [ ] **Step 1: Write failing test** `tests/test_docs.py`:
  - every `make <target>` in `docs/EXPLAINED.md` exists in the Makefile;
  - every `vjaudit.cli <group> <command>` mentioned exists (checked via `typer` `--help` output);
  - every path written as `src/vjaudit/...` exists.
- [ ] **Step 2: Run** it. Expected: FAIL (file missing).
- [ ] **Step 3: Write** `EXPLAINED.md` and update the GUIDE and README.
- [ ] **Step 4: Run** `make sanity`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "docs: EXPLAINED.md (idea, code, Mac runbook); guide profile section"`
