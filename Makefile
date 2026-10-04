.PHONY: sanity test
CI_MARKERS = not gpu and not api and not legacy and not hf

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
