"""Quick check of each local judge: one T-VCG and one T-RUB call on a fixed example (run this first on the Mac)."""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Callable

from .factory import make_client
from .parse import parse_vcg, parse_yes_no
from .readout import readout_first_token, readout_vcg
from .templates import render_rub, render_vcg

SMOKE_EXAMPLE = ("what is the man doing?", "cooking", "he is cooking food")


def smoke_judges(profile, client_factory: Callable = make_client) -> list[dict]:
    """One row per (local judge, template). A fresh temporary cache, so every smoke run really calls the model."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in profile.judges:
            if cfg.backend != "transformers":
                continue
            client = client_factory(cfg, Path(tmp))
            for template, render, max_tokens in (("t-vcg", render_vcg, 64), ("t-rub", render_rub, 1)):
                t0 = time.perf_counter()
                row = {"judge": cfg.id, "template": template, "verdict": None, "margin": None, "top_tokens": []}
                try:
                    resp = asyncio.run(client.complete(render(*SMOKE_EXAMPLE), max_tokens=max_tokens, temperature=0,
                                                       logprobs=True, top_logprobs=20))
                    choice = resp["choices"][0]
                    text, content = choice["message"]["content"], choice["logprobs"]["content"]
                    if template == "t-vcg":
                        verdict, _, _ = parse_vcg(text)
                        margin, top = readout_vcg(content)
                    else:
                        verdict, _ = parse_yes_no(text)
                        margin, top = readout_first_token(content)
                    row.update(verdict=verdict, margin=margin, top_tokens=top)
                except Exception as e:  # noqa: BLE001 — report and continue with the next call
                    row["verdict"] = f"ERROR: {e}"
                row["seconds"] = round(time.perf_counter() - t0, 2)
                rows.append(row)
    return rows
