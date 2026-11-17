"""Download images in parallel with retries, and check that each file decodes."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

log = logging.getLogger(__name__)

_USER_AGENT = "graph-retrieval-research/1.0"
_MAX_BACKOFF_S = 10


@dataclass(frozen=True)
class ImageJob:
    """One image to fetch."""

    image_id: int
    url: str


@dataclass
class DownloadReport:
    """Outcome of a download run."""

    requested: int = 0
    downloaded: int = 0
    already_present: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "requested": self.requested,
            "downloaded": self.downloaded,
            "already_present": self.already_present,
            "failed_count": len(self.failed),
            "failed": sorted(self.failed, key=lambda f: f["image_id"]),
        }


def image_path(out_dir: Path, image_id: int) -> Path:
    """Where an image is stored."""
    return out_dir / f"{image_id}.jpg"


def is_valid_image(path: Path) -> bool:
    """True when the file exists, is non-empty and Pillow can decode its header."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def download_images(
    jobs: Iterable[ImageJob], out_dir: Path, *, workers: int, retries: int, timeout_s: float
) -> DownloadReport:
    """Fetch every job not already on disk as a valid image; failures are reported, not raised."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    report = DownloadReport(requested=len(jobs))
    pending = [job for job in jobs if not is_valid_image(image_path(out_dir, job.image_id))]
    report.already_present = len(jobs) - len(pending)
    log.info("images: %d requested, %d already present, %d to fetch",
             report.requested, report.already_present, len(pending))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, job, out_dir, retries, timeout_s): job for job in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            error = future.result()
            if error is None:
                report.downloaded += 1
            else:
                job = futures[future]
                report.failed.append({"image_id": job.image_id, "url": job.url, "error": error})
            if done % 500 == 0:
                log.info("fetched %d/%d (failed %d)", done, len(pending), len(report.failed))
    return report


def _fetch_one(job: ImageJob, out_dir: Path, retries: int, timeout_s: float) -> str | None:
    """Return None on success, else the last error message. Writes via a temp file."""
    target = image_path(out_dir, job.image_id)
    tmp = target.with_suffix(".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = Request(job.url, headers={"User-Agent": _USER_AGENT})
            with urlopen(request, timeout=timeout_s) as response:
                tmp.write_bytes(response.read())
            if not is_valid_image(tmp):
                raise OSError("downloaded file is not a decodable image")
            tmp.replace(target)
            return None
        except OSError as exc:  # URLError and socket timeouts are OSError subclasses
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, _MAX_BACKOFF_S))
    tmp.unlink(missing_ok=True)
    return last_error
