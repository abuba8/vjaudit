.PHONY: sanity test
CI_MARKERS = not gpu and not api and not legacy and not hf

sanity:
	uv run pytest -m "$(CI_MARKERS)" -q

test: sanity
