# vjaudit

Auditing and repairing LLM-judged open-ended VideoQA — Stage 1 code: prior-pipeline integrity check
(SP0), benchmark loading and sampling, video-LLM generation, judging (GPT-3.5 before the 2026-10-23
shutdown, or small local judges), labelling with Claude, and a judge-vs-reference report.

**New here? Start with [docs/EXPLAINED.md](docs/EXPLAINED.md)** — the idea and the code in plain
language, plus the step-by-step Mac runbook.
**GPU server and file-by-file reference: [docs/GUIDE.md](docs/GUIDE.md).**

```bash
uv python install 3.11 && uv sync && make sanity      # CPU tests
bash scripts/download/download_all.sh                 # datasets (see the guide for sizes)

# On a 32 GB Apple-Silicon Mac (no GPU server, no OpenAI key):
uv sync --extra mac
make data-check PROFILE=mac && make data-sample PROFILE=mac && make pin PROFILE=mac && make smoke PROFILE=mac
```

Spec: `docs/superpowers/specs/` · Plan: `docs/superpowers/plans/` · Decisions: `docs/decisions.md`
