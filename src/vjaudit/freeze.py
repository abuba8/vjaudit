"""Freeze Stage-1 artifacts: sorted exports + a deterministic tarball of the API cache + manifest."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .store import ArtifactStore

KINDS = ("item", "generation", "judgecall")


@dataclass(frozen=True)
class ManifestEntry:
    kind: str
    path: str
    n_records: int
    sha256: str


def _tar_cache(cache_dir: Path, out: Path) -> tuple[int, str]:
    files = sorted(p for p in Path(cache_dir).rglob("*.json") if p.is_file())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            data = f.read_bytes()
            info = tarfile.TarInfo(str(f.relative_to(cache_dir)))
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    with gzip.GzipFile(out, "wb", mtime=0) as gz:
        gz.write(buf.getvalue())
    return len(files), hashlib.sha256(out.read_bytes()).hexdigest()


def freeze(store: ArtifactStore, cache_dir: Path, out_dir: Path,
           manifest: Path | None = None) -> list[ManifestEntry]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for kind in KINDS:
        path = out_dir / f"{kind}.jsonl"
        digest = store.export_sorted(kind, path)
        n = sum(1 for _ in path.open(encoding="utf-8"))
        entries.append(ManifestEntry(kind, str(path), n, digest))
    tar_path = out_dir / "api_cache.tar.gz"
    n, digest = _tar_cache(Path(cache_dir), tar_path)
    entries.append(ManifestEntry("api_cache", str(tar_path), n, digest))
    if manifest is not None:
        Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest).write_text("".join(json.dumps(asdict(e)) + "\n" for e in entries), encoding="utf-8")
    return entries
