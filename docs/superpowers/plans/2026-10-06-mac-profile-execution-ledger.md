# Mac profile — execution ledger (rulings, fixes, deferred items)
Spec: docs/superpowers/specs/2026-10-06-mac-profile-design.md (read). Branch: mac-profile (from stage1 @ fada471).
Setup: Ruling: plain branch `mac-profile` in the existing clone instead of a separate worktree — the clone is a sandbox copy used only for this work and stage1 is not a main branch — cost if wrong: none (branch is isolated).

## Pre-flight (shared interfaces)
| producer → consumer | produced | consumed | finding |
|---|---|---|---|
| T1 → T2 | GridCfg.frame_max_side, family "llava-onevision", Profile | cli_gen frame cache, LlavaOneVisionAdapter | consistent |
| T1 → T3 | JudgeCfg.backend/revision/chat_template_kwargs | HFCausalBackend, make_client | consistent |
| T1 → T3 | cli_gen pin-revisions default = profile generators.yaml | T3 extends to judges.yaml | consistent (T3 adds judges after generators) |
| T3 → T4 | make_client(cfg, cache_dir, backend=None, allow_unpinned=False) | capture CLI | consistent |
| T1 → T4 | Profile.generators | cli_judge._groups | consistent (T1 replaces CONFIGS read; T4 changes ablation-empty behaviour) |
| T5 → T6 | Label records; unit_id = short_id(item_id, answer_norm) | build_tables label join | consistent |
| T2/T3/T5/T6 → T7 | gen smoke, judge smoke, gen pin-revisions, annotate export, report summary | Makefile targets | consistent |
| T7 → T8 | Makefile targets | EXPLAINED.md + test_docs | consistent |
Task 1: Ruling: report defaults are per profile, docs/<stem>_<profile>.md, server included (data_availability, samples, capture) — profiles must not overwrite each other's reports — cost if wrong: GUIDE paths change (updated in Task 8).
Task 1: Ruling: a --profile value is a folder only if it contains a path separator; otherwise a name under configs/profiles — a cwd folder named "mac" must not hijack the profile — cost if wrong: none.
Task 1: Ruling: golden_render reads configs/profiles/server/generators.yaml directly instead of load_profile("server") — golden prompts need no machine paths.yaml — cost if wrong: none.
Task 1: Ruling: mac generators use `dtype: auto` (= device default per spec §3); explicit dtypes still honoured by TransformersEngine (Task 2) — cost if wrong: one config field.
Task 1: Ruling: freeze script writes artifacts/manifest_<profile>.jsonl for non-server profiles; server keeps artifacts/manifest.jsonl — cost if wrong: none.
Task 1: Ruling: mac stage requests carry n: 1 like the server stages; test asserts that — cost if wrong: none.
Task 1: Ruling: PROFILE/VJAUDIT_PROFILE Makefile plumbing deferred to Task 7 (which owns the Makefile) — cost if wrong: none.
Task 1: complete (commits fada471..e855767, tests: make sanity → 211 passed, 14 deselected in 3.96s)
Task 2: Ruling: TransformersEngine passes do_sample_frames=False with videos (frames are already sampled by the frame cache) and casts float inputs to the model dtype via BatchFeature.to(device, dtype=...) — spec §3 lists only the bare processor call; without these a processor may re-sample our 8 frames or feed fp32 pixels to an fp16 model — cost if wrong: a processor rejects the kwarg; `gen smoke` shows it at once.
Task 2: Ruling: smoke_generators catches a failing generator, emits an "ERROR: ..." row and continues; `gen smoke` exits 1 if any row is an error — one broken model must not hide the others' results — cost if wrong: none.
Task 2: Ruling: `gen rerun-check` also goes through make_engine/make_renderer (not in the plan) — keeps the Mac profile usable for the reproducibility check — cost if wrong: none.
Task 2: Ruling: added CLI tests test_gen_smoke_cli_without_items_says_sample_first / test_gen_smoke_cli_prints_rows for the brief's "run make data-sample first" behaviour — cost if wrong: none.
Task 2: complete (commits e855767..73b9690, tests: make sanity → 228 passed, 14 deselected in 4.12s)
Task 3: Ruling: make_client caches local judges under <api_cache>/<judge id>/<revision> — the request hash holds the repo but not the revision, so a re-pinned model would otherwise reuse old verdicts — cost if wrong: none (extra folder level).
Task 3: Ruling: sampling (temperature > 0 or absent) uses top_p=1.0, top_k=0 to mirror the API's untruncated default sampling; greedy clears the model's generation_config sampling fields — cost if wrong: S5-style n=3 stages differ from a model's tuned defaults (no Mac stage uses sampling).
Task 3: Ruling: my first margin expectation (2.0) in test_local_transport_vcg_margin_readable / test_smoke_judges_rows was wrong — the readout correctly sums every yes/no spelling present in top_logprobs; expectation now log(e^5+2)-log(e^3+2) — cost if wrong: none.
Task 3: Ruling: `gen pin-revisions` reads hf_resolver from the module at call time (testable) and pins generators.yaml then judges.yaml; `--path` pins one file — cost if wrong: none.
Task 3: Ruling: `judge smoke` exits 1 when any row is an error, like `gen smoke` — cost if wrong: none.
Task 3: complete (commits 73b9690..dfcde72, tests: make sanity → 244 passed, 14 deselected in 4.16s)
Task 4: Ruling: `_groups` needed no change — `judged - ablation` already equals all systems when no generator has frame_ablation; test_groups_without_ablation_is_all_systems pins it (passes as a characterization test) — cost if wrong: none.
Task 4: Ruling: select_stages raises on unknown stage names (CLI exit 2 with the list) instead of silently ignoring them — a typo like M7 must not run nothing quietly — cost if wrong: none.
Task 4: Ruling: JudgeCall.judge_rev = cfg.revision for local judges (cfg.model for API judges); capture report title generalised to "Judge capture report"; dry run skips cost estimation for local judges (0.00) — cost if wrong: none.
Task 4: Ruling: my dry-run test first expected M6 to have jobs; build_jobs correctly skips main-set calls already scheduled by M1/M2 for items in both sets (run 2 reuses run 1) — test now asserts "M5: 0 jobs" — cost if wrong: none.
Task 4: complete (commits dfcde72..9250099, tests: make sanity → 251 passed, 14 deselected in 4.17s)
Task 5: Ruling: re-running `annotate export` writes only units not yet in the folder's manifest, as new batches numbered after the existing ones (manifest appended) — re-planning all units would renumber batches and orphan already-imported labels — cost if wrong: a question's later units can land in a later batch.
Task 5: Ruling: build_units skips style "mc" generations (letters, not open-ended answers) — cost if wrong: none.
Task 5: Ruling: Markdown cells are one line with escaped pipes and "(empty answer)" for blanks — Claude must see every unit unambiguously — cost if wrong: none.
Task 5: Ruling: import also rejects a unit given twice with different labels, and a reply with no unit_id header line; CLI errors are prefixed "<file> row <n>" — cost if wrong: none.
Task 5: Ruling: `annotate import` picks the manifest by source (claude-manual* → claude/, human:* → human/); `--out-dir` (default annotation/) is the base; /annotation/ is gitignored (it holds dataset text) — cost if wrong: none.
Task 5: Ruling: my first export tests assumed units[0] was the "dog" unit; units sort by (item_id, unit_id) so tests now look the unit up — cost if wrong: none.
Task 5: complete (commits 9250099..7293359, tests: make sanity → 275 passed, 14 deselected in 4.51s)
Task 6: Ruling: "first label source" = alphabetically first source (claude-manual before human:*); the agreement table names it — cost if wrong: agreement shown against one source only; per-source accuracy columns still show every source.
Task 6: Ruling: results Markdown splits each benchmark into an accuracy table (configs + sources) and a separate agreement table (gap/κ/FPR/FNR) for readability; visual gain is a bullet list per benchmark/system — spec §7 fixes the content, not the layout — cost if wrong: layout only.
Task 6: Ruling: report ignores style "mc" generations and only uses label task "ref" — cost if wrong: none.
Task 6: complete (commits 7293359..f01eabc, tests: make sanity → 286 passed, 14 deselected in 4.76s)
Task 7: Ruling: added `annotate-import` (FILES=..., SOURCE ?= claude-manual) and `annotate-status` targets and an ALLOW_UNVERIFIED=1 flag for the judge targets — the runbook needs a one-line import and the unverified-prompt escape hatch — cost if wrong: none.
Task 7: Ruling: `make smoke` ignores a gen-smoke failure (`-` prefix) so the judge smoke still runs; ERROR rows show what failed — cost if wrong: make's exit code hides a generator failure.
Task 7: Ruling: `uv lock` resolved gpu + mac together (torch 2.10.0 with a macOS arm64 wheel; vllm 0.18.1 Linux-only), so no [tool.uv] conflicts entry — cost if wrong: none.
Task 7: complete (commits f01eabc..8f09ca3, tests: make sanity → 290 passed, 16 deselected in 4.76s)
Task 8: Ruling: test_docs.py checks README.md and GUIDE.md as well as EXPLAINED.md, also checks configs/scripts/tests paths, and matches `make <target>` only at line start or after a backtick (prose like "make sure" is not a target) — stale GUIDE config paths were exactly what it caught — cost if wrong: none.
Task 8: Ruling: example outputs in EXPLAINED.md are labelled as illustrations; pinned SHAs shown as placeholders; NExT-OE test count (9,178) read from the user's parquet — cost if wrong: none.
Task 8: Ruling: added a dated docs/decisions.md entry for the Mac profile — cost if wrong: none.
Task 8: complete (commits 8f09ca3..6396ab0, tests: make sanity → 299 passed, 16 deselected in 4.63s)
Final review: opus reviewer, verdict "With fixes"; 0 Critical, 7 Important, 10 Minor.
Final: Ruling: regraded Minor 1 (Makefile ignored VJAUDIT_PROFILE though GUIDE documents it), Minor 2 (two local judges could run concurrently on MPS at a stage boundary), Minor 5 (printed import hint lacked --profile and fails when followed), Minor 6 (doc'd recovery for cut-off replies fails on the partial row) to Important — each makes a documented path fail for a person following it — cost if wrong: four small extra fixes.
Final: fixed sanity fails after `export PROFILE=mac` — RED by running suite with VJAUDIT_PROFILE=mac (7 failed) → autouse fixture → GREEN, suite 299/299
Final: fixed frame size missing from generation ID — test_frame_size_is_part_of_the_generation_id RED→GREEN (server IDs unchanged at 768), suite 300/300
Final: fixed report mixing verdicts from older prompt versions — test_verdicts_from_an_older_template_version_are_left_out_and_counted RED→GREEN
Final: fixed report hiding partial judge coverage — test_partial_coverage_is_shown_in_the_cell RED→GREEN, suite 302/302
Final: fixed unfenced replies with closing prose / spaced headers / extra fields — test_parse_unfenced_reply_ignores_trailing_prose, test_parse_header_with_spaces_after_commas, test_row_with_too_many_fields_is_an_error RED→GREEN
Final: fixed import hint without --profile — test_export_import_status_round_trip (hint assertion) RED→GREEN
Final: fixed greedy decoding inheriting generation_config (repetition penalty) — test_engine_greedy_ignores_model_generation_defaults, test_judge_generate_kwargs_greedy_and_sampling RED→GREEN
Final: fixed concurrent local judges — test_local_judges_never_run_at_the_same_time RED→GREEN
Final: fixed Makefile ignoring VJAUDIT_PROFILE — test_makefile_profile_precedence RED→GREEN
Final: fixed silent invisibility of pre-profile records — test_cli_warns_about_records_in_the_old_unprofiled_layout RED→GREEN + GUIDE move steps; suite 310/310
Final: fixed doc recovery for cut-off Claude replies (delete partial last row) — doc-only, no test
Final: Ruling: float16 on MPS kept (spec fixes it); generators can set dtype, judges cannot yet (deferred minor) — cost if wrong: a judge misbehaving in fp16 needs a code change.
Final: Ruling: Qwen fps assumptions for pre-sampled frames (2 fps / 24 fps metadata) left as is — affects answer quality uniformly on the Mac; spec fixes the bare processor call — cost if wrong: weaker temporal answers from Qwen models.
Final: Ruling: no PYTORCH_ENABLE_MPS_FALLBACK handling — `gen smoke` reveals unsupported ops in minutes — cost if wrong: one troubleshooting round on the Mac.
Final: Ruling: label_id ignores task — vid labelling is out of scope — cost if wrong: must add task to label_id before vid labels from the same source.
Final: Ruling: judge_call_id excludes the judge revision (Stage-1 ID design); runbook pins before judging — cost if wrong: re-pinning a judge mid-project keeps old verdicts (judge_rev shows which).
Final: Ruling: partly filled human sheets are rejected whole, Excel-format CSVs fail safely — spec requires valid labels; human path is secondary on the Mac — cost if wrong: an annotator re-saves as UTF-8 CSV.
Final: Ruling: visual gain computed over whatever items each system covers — only differs when videos fail to decode, visible in coverage — cost if wrong: small bias in a rare case.
Final: Ruling: no golden prompt for llava-onevision; rerun-check does not close its engine; CPU-then-MPS load — plan scope / one-off command / fits in 32 GB — cost if wrong: none material.
Final: minor (deferred): agreement uses the alphabetically first label source (claude-manual beats claude-manual-v2 / human:*); add a --reference option
Final: minor (deferred): freeze exports only item/generation/judgecall (no unit/label); artifacts/manifest_mac.jsonl not excepted in .gitignore
Final: minor (deferred): no test of the BatchFeature dtype cast (do_sample_frames now tested)
Final: minor (deferred): JudgeCfg has no dtype override for local judges
Final: minor (deferred): gen mc-report / scripts/throughput.py report paths are not per profile; stale docstrings in cli_judge.py and availability.py
Final: minor (deferred): an MPS out-of-memory while loading surfaces as "could not load ..." without the OOM hint
