# vjaudit

Auditing and repairing LLM-judged open-ended VideoQA — Stage 1 code: prior-pipeline integrity check
(SP0), benchmark loading and sampling, video-LLM generation, judging (GPT-3.5 before the 2026-10-23
shutdown, or small local judges), labelling with Claude, and a judge-vs-reference report.

**New here? Start with [docs/EXPLAINED.md](docs/EXPLAINED.md)** — the idea and the code in plain
language, plus the step-by-step Mac runbook.
**GPU server and file-by-file reference: [docs/GUIDE.md](docs/GUIDE.md).**

## Install

Python 3.11, 3.12 or 3.13. Use either a conda environment or uv.

**Option A: conda** (for example an existing environment called `dl`)

```bash
conda activate dl
pip install -e ".[mac]" pytest pytest-asyncio     # on a Mac; use ".[gpu]" on the GPU server
make sanity                                       # CPU tests; must end with "passed" and no "failed"
```

`pip install` may upgrade `torch` and `transformers` inside `dl` (the code needs `transformers>=4.57`).
While a conda environment other than `base` is active, every `make` command uses that environment's
`python` and `pytest`; uv is not needed.

**Option B: uv**

```bash
uv python install 3.11
uv sync --extra mac                               # on a Mac; --extra gpu on the GPU server
make sanity
```

## First commands on a Mac

```bash
conda activate dl                                 # skip with uv
export VJAUDIT_PROFILE=mac                        # make and CLI commands then use the Mac set-up
make data-check
make data-sample
make pin
make smoke
```

Then follow Part 5 of [docs/EXPLAINED.md](docs/EXPLAINED.md), or the command list in
[WORKFLOW.md](WORKFLOW.md), which also traces each step through the code.

**Prefer a notebook?** [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) runs the same pipeline in one
self-contained, explained Jupyter notebook (see [notebooks/README.md](notebooks/README.md)). The datasets come from
`bash scripts/download/download_all.sh` (see the guide for sizes).

Spec: `docs/superpowers/specs/` · Plan: `docs/superpowers/plans/` · Decisions: `docs/decisions.md`
