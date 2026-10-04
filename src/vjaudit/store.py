"""Append-only, content-addressed JSONL artifact store (R-GLB-05).

Layout: ``<root>/<kind>/part-<pid>-<n>.jsonl``; each line ``{"key": ..., "record": {...}}``.
Writes are flushed and fsynced per line. Each process writes its own part file, so
resuming after a crash never edits an existing file. ``export_sorted`` gives the
canonical, order-independent view used to compare runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from .hashing import canonical_json, config_hash, git_sha

log = logging.getLogger(__name__)


def stamp(record_dict: dict, cfg: dict) -> dict:
    """Return ``record_dict`` plus config_hash, git_sha and a UTC created_at."""
    return {
        **record_dict,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class ArtifactStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._keys: dict[str, set[str]] = {}
        self._handles: dict[str, object] = {}
        for kind_dir in sorted(p for p in self.root.iterdir() if p.is_dir() and ".archived-" not in p.name):
            self._keys[kind_dir.name] = {row["key"] for row in self._read_kind(kind_dir.name)}

    # -- reading -------------------------------------------------------------
    def _parts(self, kind: str) -> list[Path]:
        return sorted((self.root / kind).glob("part-*.jsonl"))

    def _read_kind(self, kind: str) -> Iterator[dict]:
        for part in self._parts(kind):
            lines = part.read_text(encoding="utf-8").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            for i, line in enumerate(lines):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    if i == len(lines) - 1:
                        log.warning("ignoring truncated last line in %s", part)
                        continue
                    raise ValueError(f"corrupt line {i + 1} in {part}") from e

    def iter(self, kind: str) -> Iterator[dict]:
        """Yield stored records (the ``record`` payloads) of one kind, first copy per key."""
        seen: set[str] = set()
        for row in self._read_kind(kind):
            if row["key"] not in seen:
                seen.add(row["key"])
                yield row["record"]

    def has(self, kind: str, key: str) -> bool:
        return key in self._keys.get(kind, set())

    # -- writing -------------------------------------------------------------
    def _handle(self, kind: str):
        if kind not in self._handles:
            d = self.root / kind
            d.mkdir(parents=True, exist_ok=True)
            n = 0
            while (d / f"part-{os.getpid()}-{n}.jsonl").exists():
                n += 1
            self._handles[kind] = open(d / f"part-{os.getpid()}-{n}.jsonl", "a", encoding="utf-8")
        return self._handles[kind]

    def put(self, kind: str, record: BaseModel, key: str) -> bool:
        """Append ``record`` under ``key``; return False (and write nothing) if the key exists."""
        keys = self._keys.setdefault(kind, set())
        if key in keys:
            return False
        f = self._handle(kind)
        f.write(json.dumps({"key": key, "record": record.model_dump(mode="json")}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        keys.add(key)
        return True

    def archive(self, kind: str) -> Path:
        """Move a derived kind aside (``<kind>.archived-<UTC>``) so it can be rebuilt; nothing is deleted."""
        f = self._handles.pop(kind, None)
        if f is not None:
            f.close()
        stamp_ = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst = self.root / f"{kind}.archived-{stamp_}"
        (self.root / kind).rename(dst)
        self._keys.pop(kind, None)
        return dst

    def close(self) -> None:
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    # -- export --------------------------------------------------------------
    def export_sorted(self, kind: str, out: Path | str) -> str:
        """Write records sorted by key without ``created_at``; return the file's sha256."""
        rows = {}
        for row in self._read_kind(kind):
            rows.setdefault(row["key"], row["record"])
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for key in sorted(rows):
                rec = {k: v for k, v in rows[key].items() if k != "created_at"}
                f.write(canonical_json({"key": key, "record": rec}) + "\n")
        return hashlib.sha256(out.read_bytes()).hexdigest()
