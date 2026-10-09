# SDD ledger — plan: docs/superpowers/plans/2026-10-08-pipeline-notebook.md
Spec: docs/superpowers/specs/2026-10-08-pipeline-notebook-design.md (read). Branch: mac-profile (v2) @ 4317fbe. Execution: Native (user: "go for it, create the notebook").
Setup: Ruling: continue on branch mac-profile instead of a new worktree/branch — the notebook is part of v2, which the user keeps as its own folder on that branch — cost if wrong: none.
Setup: Ruling: no separate plan-review gate — user approved the spec and said "go for it, create the notebook" — cost if wrong: plan changes come later as fixes.

## Pre-flight (shared interfaces)
| producer → consumer | produced | consumed | finding |
|---|---|---|---|
| T1 → T2–T6 | settings(), Store, short_id, normalize_answer, need | every section | consistent |
| T2 → T3 | items table (item_id, benchmark, video_id, video_path, available, in_main/in_annotated, mc_*) | run_generation | consistent |
| T3 → T4 | generations table (gen_id, item_id, system_id, output_text, answer_norm, style) | run_stages | consistent |
| T3/T4 → T5 | generations + judgecalls (judge_id, protocol_id, parsed_verdict, sample_idx) | build_units, build_tables | consistent |
| T1–T5 → T6 | whole notebook with parameters/lib tags | execution tests | consistent |
Task 1: Ruling: settings split into two cells — the `parameters` cell holds only the everyday switches; the MAC/SERVER dictionaries live in the next cell (tagged lib) — keeps the cell you edit short; spec said "in the cell" — cost if wrong: one scroll.
Task 1: Ruling: model `revision: None` (latest) by default in the notebook; the commit id actually loaded is recorded with every row (section 8), and the markdown says how to pin — notebook has no `make pin` — cost if wrong: answers may come from a newer model version unless pinned.
Task 1: Ruling: the notebook source is authored in percent format in the session scratchpad and built with nbformat; only the .ipynb ships — cost if wrong: a future edit converts the .ipynb back first.
Task 1: Ruling: the "13 sections" check moves to Task 5 (whole-notebook property); Task 1 keeps "every header is in the table of contents" — cost if wrong: none.
Task 1: complete (commits 4317fbe..7d8210e, tests: make sanity → 327 passed, 16 deselected in 6.07s)
Task 2: Ruling: items.jsonl stores only the main sample (annotated ⊂ main ⊂ pool), not every pool row with flags like the CLI — smaller file, nothing downstream needs the unsampled rows (they stay in pool.jsonl) — cost if wrong: none.
Task 2: complete (commits 7d8210e..3a1be7f, tests: make sanity → 334 passed, 16 deselected in 6.20s)
Task 3: Ruling: generation IDs use the configured revision (None = latest), not the resolved commit — resolving needs the network and would change every ID offline; the actual commit is stored per row (`generator_rev`) — cost if wrong: answers from two model versions can share an ID if the Hub updates mid-run (visible in generator_rev).
Task 3: Ruling: the multiple-choice anchor is a toggle cell (`RUN_MC_ANCHOR = False`) at the end of section 8, not a section-2 setting — it is optional and server-oriented — cost if wrong: none.
Task 3: Ruling: my first frame-cache test used a 30-frame synthetic video; the test helper's pixel values overflow above 25 frames, so the test now uses 20 — cost if wrong: none.
Task 3: complete (commits 3a1be7f..dce769f, tests: make sanity → 344 passed, 16 deselected in 6.51s)
Task 4: Ruling: every judge returns the same dict from .complete(messages, request) — {choices: [{text, content}], model, cost, cached} — instead of the CLI's OpenAI-shaped transport; one interface for fake, local and API judges keeps the notebook short — cost if wrong: none (equivalence tests cover parse and margin).
Task 4: Ruling: the API spend estimate uses characters/3 (no tiktoken) and caches by sha256 of the canonical request — the notebook avoids the tokenizer download — cost if wrong: a slightly loose pre-call estimate; actual spend comes from usage.
Task 4: Ruling: judge_call_id keeps the package formula short_id(gen_id, judge_id, protocol_id, "none", "text", k) — same IDs as the CLI for the same inputs — cost if wrong: none.
Task 4: complete (commits dce769f..1c6eaf4, tests: make sanity → 369 passed, 16 deselected in 6.43s)
Task 5: Ruling: section 10 imports every *.txt / *.csv in annotation/claude/replies/ on each run (no file list to edit) — re-importing is idempotent — cost if wrong: a stray file in replies/ shows up as a listed problem.
Task 5: Ruling: the report notes unverified judge prompts by reading configs/prompts/judge/source.yaml (a prompt-provenance data file, read with PyYAML) — spec allows reading configs/prompts — cost if wrong: none.
Task 5: complete (commits 1c6eaf4..8f4f05e, tests: make sanity → 386 passed, 16 deselected in 6.46s)
Task 6: Ruling: the execution tests passed on first run (the notebook already existed); I checked they can fail by executing with a non-existent DATA_ROOT, which stops with the expected FileNotFoundError — cost if wrong: none.
Task 6: Ruling: added `ipywidgets` to the `notebook` extra and silenced tqdm's fallback warning; progress bars now name system · benchmark — found while reading the dry-run output — cost if wrong: none.
Task 6: complete (commits 8f4f05e..06c3c47, tests: make sanity → 391 passed, 16 deselected in 14.98s)
Task 6: Ruling: fixed my own README edit that had attached the datasets sentence to the notebook paragraph — cost if wrong: none.

## Final review
Final: review (opus, range 4317fbe..f7c8491) — "Ready to merge? With fixes": 1 Critical, 4 Important, 13 Minor.
Final: fixed Critical 1 (restart path: sections 8, 9, 11 raised NameError after "run sections 2 and 3, then any section") — section 3 now runs the definition cells of sections 4–11 from the saved notebook (`load_definitions`), and says to save after editing a definition — test_after_a_restart_sections_2_and_3_then_any_section[4..11] RED (8, 9, 11) → GREEN, suite 407/407.
Final: fixed Important 2 (crash safety) — Store removes a cut-off last line before appending (a complete row missing only its newline is kept), and a corrupt row names its file and line; API cache entries are written to a temp file and renamed, unreadable entries are skipped and asked again — test_store_repairs_a_cut_off_last_line, test_store_names_file_and_line_of_a_broken_row, test_api_cache_survives_a_broken_file RED→GREEN, suite 407/407.
Final: fixed Important 3 (a changed set-up mixed old and new answers in sections 9–11) — `current_generations` keeps answers whose (benchmark, system, config hash) match the current profile; used by coverage, run_stages, the section 10 export and the section 11 report; section 8 prints how many older answers are kept but unused; new row in section 12 — test_after_a_setup_change_only_current_answers_are_used RED→GREEN, suite 407/407.
Final: fixed Important 4 (a model that failed to load stopped the cell, spec §6) — the error is printed as "<model>: could not load — …", the other models run, and the cell ends with a NOT DONE line — test_a_model_that_fails_to_load_is_skipped_and_reported RED→GREEN, suite 407/407.
Final: fixed Important 5 (dry run not ~1 minute on real data; video check saved only at the end) — a dry run checks 20 random questions per benchmark (`check_limit`), stand-in models read no frames (`needs_frames`), the video check is saved every 200 questions and resumes, and the time text is corrected (about 2,500 videos for about 30,000 questions; dry run 1–2 minutes) — test_video_check_saves_in_chunks_and_resumes, test_dry_run_checks_only_a_few_questions_per_benchmark, test_stand_in_model_decodes_no_frames RED→GREEN, suite 407/407.
Final: minor fixed — `%pip` instead of `!pip` in the commented install cells (installs into the kernel's environment).
Final: minor fixed — `OpenAI(max_retries=0)`: the client's own retries no longer stack with the notebook's six.
Final: minor fixed — section 2 text said "the cell after it" where three cells follow.
Final: minor fixed — section 10 says to save replies as plain text (TextEdit's .rtf default), and lists files in replies/ that it does not read.
Final: minor (deferred): a 401 (bad API key) records every call of the stage as failed instead of stopping the stage.
Final: minor (deferred): the APIJudge constructor reads every cache file to add up spending (slow with ~30,000 cached replies).
Final: minor (deferred): API judge calls run one at a time (~30 h for the server grid); add a time estimate or a thread pool.
Final: minor (deferred): margins that are NaN are written as bare `NaN` in JSONL (Python reads it; strict JSON readers do not).
Final: minor (deferred): a second-pass import under `claude-manual-v2` re-reads the same replies/ folder, so first-pass files are imported under the new name too.
Final: minor (deferred): the human-sheet import note says to point ANNOTATION to the human folder but gives no cell for it.
Final: minor (deferred): LocalJudge has no dtype override and no out-of-memory hint.
Final: minor (deferred): no early check that REPO/configs/prompts exists (fails later with FileNotFoundError naming the file).
Final: minor (deferred): the dry-run and real results folders do not share the video check (the real run checks the videos again).
Final: Ruling: test_undecodable_video_is_skipped_and_listed now uses an engine that needs frames, since the stand-in no longer reads them — cost if wrong: none.
Final: Ruling: the judging test fixture gives its synthetic answers the current config hash, since judging now uses only current-set-up answers — cost if wrong: none.
Final: Ruling: build_tables/write_results take an optional `gens` (default: every stored answer) and the report cell passes current_generations — keeps them testable without a profile — cost if wrong: a hand call without `gens` can mix set-ups.
Final: Ruling: load_definitions reads the saved .ipynb, so unsaved edits to definition cells are not seen after a restart; section 3 says to save first — cost if wrong: an older definition runs until that section's own cell runs.
Final: Ruling: vLLM video input for Qwen3-VL (frames tuple without metadata) — not judged by the reviewer; same format as the package's server path, which runs on the server only — cost if wrong: server generation for Qwen3-VL fails or uses default timestamps; checked on the first server run.
Final: Ruling: vLLM memory between models (del + empty_cache) — not judged; same as the package — cost if wrong: the server needs one model per kernel.
Final: Ruling: Qwen3-VL timestamps assume the processor default fps for pre-sampled frames — not judged; same as the package's hf path — cost if wrong: slightly wrong time stamps in the prompt for Qwen3-VL.
Final: Ruling: float16 by default on MPS — not judged; covered by the Mac-profile ruling and the dtype override in section 8/12 — cost if wrong: set bfloat16.
Final: Ruling: gen_id from the config hash, stage order, unpinned revisions, chars/3 estimate, items-only-main, the separate profile cell — each already covered by an earlier Ruling in this ledger.
Final: Ruling: the spec §4 section table is not repeated in the notebook; its table of contents and section headers carry it — cost if wrong: none.
Final: complete (fix pass on f7c8491, tests: make sanity → 407 passed, 16 deselected)
