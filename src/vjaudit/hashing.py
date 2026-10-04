"""Canonical hashing helpers shared by every artifact (contract R-GLB-05)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` deterministically: sorted keys, no spaces, unicode kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def config_hash(obj: Any) -> str:
    """64-hex sha256 of the canonical JSON of a config dict."""
    return sha256_hex(canonical_json(obj))


def short_id(*parts: str) -> str:
    """16-hex id = sha1("|".join(parts))[:16] (contract §8.2 gen_id formula)."""
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def git_sha() -> str:
    """Current commit SHA, or "nogit" when not inside a git work tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return out.stdout.strip() or "nogit"
