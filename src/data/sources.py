"""Fetch raw files and keep a sha256 manifest so corrupted or swapped files are caught."""
from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from src.core.exceptions import ChecksumMismatchError, DataError
from src.data.config import SourceSpec
from src.data.json_io import read_json, write_json

log = logging.getLogger(__name__)

_USER_AGENT = "graph-retrieval-research/1.0"
_CHUNK_BYTES = 1 << 20
_DOWNLOAD_TIMEOUT_S = 120


@dataclass(frozen=True)
class ManifestEntry:
    """What we know about one raw file on disk."""

    name: str
    path: str
    url: str
    bytes: int
    sha256: str
    recorded_at: str


def sha256_of(path: Path) -> str:
    """Stream a file through sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Read the manifest keyed by source name; empty when the file does not exist."""
    if not path.exists():
        return {}
    return {row["name"]: ManifestEntry(**row) for row in read_json(path)}


def save_manifest(path: Path, entries: dict[str, ManifestEntry]) -> None:
    """Write the manifest sorted by source name."""
    write_json(path, [asdict(entries[name]) for name in sorted(entries)])


def ensure_source(spec: SourceSpec, raw_dir: Path, manifest: dict[str, ManifestEntry]) -> ManifestEntry:
    """Make sure the raw file exists and matches the manifest; fetch it when missing.

    A file already on disk but absent from the manifest is trusted once and recorded,
    which pins the files downloaded before the manifest existed.
    """
    target = raw_dir / spec.target
    if not target.exists():
        log.info("fetching %s from %s", spec.name, spec.url)
        _fetch(spec, target)
    digest = sha256_of(target)
    known = manifest.get(spec.name)
    if known is not None:
        if known.sha256 != digest:
            raise ChecksumMismatchError(
                f"{target}: sha256 {digest} differs from manifest {known.sha256}"
            )
        log.info("verified %s (%s)", spec.name, target)
        return known
    entry = ManifestEntry(
        name=spec.name,
        path=str(target),
        url=spec.url,
        bytes=target.stat().st_size,
        sha256=digest,
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    log.info("recorded %s in manifest (%d bytes)", spec.name, entry.bytes)
    return entry


def _fetch(spec: SourceSpec, target: Path) -> None:
    """Download to a temp dir, extract the archive member if any, then move into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp_dir:
        download = Path(tmp_dir) / "download"
        request = Request(spec.url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response, download.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK_BYTES)
        if spec.archive_member is None:
            download.replace(target)
            return
        with zipfile.ZipFile(download) as archive:
            member = _find_member(archive, spec.archive_member)
            with archive.open(member) as src, (Path(tmp_dir) / "member").open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK_BYTES)
        (Path(tmp_dir) / "member").replace(target)


def _find_member(archive: zipfile.ZipFile, wanted: str) -> str:
    """Match by base name, since archives often nest files in a folder."""
    matches = [n for n in archive.namelist() if Path(n).name == wanted]
    if len(matches) != 1:
        raise DataError(f"archive member '{wanted}': expected 1 match, found {matches}")
    return matches[0]
