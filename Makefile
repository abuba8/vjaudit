.PHONY: sanity test
# Which machine profile to use: configs/profiles/<PROFILE>/ (server = GPU server, mac = Apple-Silicon laptop).
PROFILE ?= server
export VJAUDIT_PROFILE = $(PROFILE)
CLI = uv run python -m vjaudit.cli
# ALLOW_UNPINNED=1 / ALLOW_UNVERIFIED=1: quick tests only (results are not paper-grade).
UNPINNED_FLAG = $(if $(ALLOW_UNPINNED),--allow-unpinned,)
UNVERIFIED_FLAG = $(if $(ALLOW_UNVERIFIED),--allow-unverified,)
CI_MARKERS = not gpu and not api and not legacy and not hf and not mac

sanity:
	uv run pytest -m "$(CI_MARKERS)" -q

test: sanity

.PHONY: data-check data-sample generate-annotated generate-main generate-mc gen-coverage throughput
SHARDS = 0 1 2

data-check:
	uv run python -m vjaudit.cli data check-availability

data-sample:
	uv run python -m vjaudit.cli data sample

# One process per GPU; generators are split round-robin across the 3 shards.
generate-annotated:
	mkdir -p logs
	for k in $(SHARDS); do CUDA_VISIBLE_DEVICES=$$k uv run python -m vjaudit.cli gen run --subset annotated --shard $$k --num-shards 3 > logs/gen-annotated-$$k.log 2>&1 & done; wait
	uv run python -m vjaudit.cli gen coverage --subset annotated

generate-main:
	mkdir -p logs
	for k in $(SHARDS); do CUDA_VISIBLE_DEVICES=$$k uv run python -m vjaudit.cli gen run --subset main --shard $$k --num-shards 3 > logs/gen-main-$$k.log 2>&1 & done; wait
	uv run python -m vjaudit.cli gen coverage --subset main

generate-mc:
	mkdir -p logs
	for k in $(SHARDS); do CUDA_VISIBLE_DEVICES=$$k uv run python -m vjaudit.cli gen run --subset main --mc --shard $$k --num-shards 3 > logs/gen-mc-$$k.log 2>&1 & done; wait
	uv run python -m vjaudit.cli gen mc-report

throughput:
	uv run python scripts/throughput.py --n 50

.PHONY: capture-gpt35
# Dry run by default (prints per-stage cost estimates). Real run: make capture-gpt35 CONFIRM=1 [STAGES=S1,S2]
capture-gpt35:
	uv run python -m vjaudit.cli judge capture $(if $(STAGES),--stages $(STAGES),) $(if $(CONFIRM),--confirm,)

.PHONY: freeze sp0-report
freeze:
	uv run python scripts/freeze_stage1.py

sp0-report:
	uv run python scripts/sp0_report.py configs/local/legacy.yaml docs/sp0_report.md

# ---- Mac profile workflow (make <target> PROFILE=mac); see docs/EXPLAINED.md -----------------------------
.PHONY: pin smoke run1 run2 judge-run1 judge-run2 annotate-export annotate-import annotate-status report
pin:
	$(CLI) gen pin-revisions

# A failing generator does not stop the judge check; look for ERROR rows.
smoke:
	-$(CLI) gen smoke $(UNPINNED_FLAG)
	$(CLI) judge smoke $(UNPINNED_FLAG)

run1:
	$(CLI) gen run --subset annotated $(UNPINNED_FLAG)
	$(CLI) gen coverage --subset annotated

run2:
	$(CLI) gen run --subset main $(UNPINNED_FLAG)
	$(CLI) gen coverage --subset main

judge-run1:
	$(CLI) judge capture --stages M1,M2,M3,M4 --confirm $(UNPINNED_FLAG) $(UNVERIFIED_FLAG)

judge-run2:
	$(CLI) judge capture --stages M5,M6 --confirm $(UNPINNED_FLAG) $(UNVERIFIED_FLAG)

annotate-export:
	$(CLI) annotate export --set annotated --for claude

# make annotate-import PROFILE=mac FILES="replies/batch_01.txt replies/batch_02.txt" [SOURCE=claude-manual]
SOURCE ?= claude-manual
annotate-import:
	$(CLI) annotate import --source $(SOURCE) $(foreach f,$(FILES),--file $(f))

annotate-status:
	$(CLI) annotate status

report:
	$(CLI) report summary
