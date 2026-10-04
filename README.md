# vjaudit

Auditing and repairing LLM-judged open-ended VideoQA — Stage 1 code: prior-pipeline integrity check
(SP0), benchmark loading and sampling, video-LLM generation, and capture of GPT-3.5 judge verdicts
before the 2026-10-23 shutdown.

**Start here: [docs/GUIDE.md](docs/GUIDE.md)** — step-by-step runbook, architecture, and a
file-by-file reference.

```bash
uv python install 3.11 && uv sync && make sanity      # CPU tests
bash scripts/download/download_all.sh                 # datasets (see the guide for sizes)
```

Spec: `docs/superpowers/specs/` · Plan: `docs/superpowers/plans/` · Decisions: `docs/decisions.md`
