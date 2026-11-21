"""Read entity images out of the MKG-W archive and keep only files that decode."""
from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
_UTF8_FLAG = 0x800


def _member_name(info: zipfile.ZipInfo) -> str:
    """Member path as UTF-8. `zipfile` decodes as cp437 when the archive lacks the UTF-8 flag."""
    if info.flag_bits & _UTF8_FLAG:
        return info.filename
    return info.filename.encode("cp437").decode("utf-8", errors="replace")


def image_members(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    """Group image files by entity folder (`<root>/<folder>/<file>`), files in name order."""
    folders: dict[str, list[zipfile.ZipInfo]] = {}
    for info in sorted(archive.infolist(), key=_member_name):
        parts = PurePosixPath(_member_name(info)).parts
        if not info.is_dir() and len(parts) == 3 and PurePosixPath(parts[2]).suffix.lower() in _IMAGE_SUFFIXES:
            folders.setdefault(parts[1], []).append(info)
    return folders


def decodes(payload: bytes) -> bool:
    """True when PIL can fully decode the bytes as an RGB image."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        # PIL raises these for truncated or mislabeled files; such a file is skipped, not fatal.
        return False
    return True


def extract_images(
    archive: zipfile.ZipFile,
    folder_of_entity: Mapping[int, str],
    members: Mapping[str, list[zipfile.ZipInfo]],
    out_dir: Path,
) -> tuple[dict[int, list[str]], list[str]]:
    """Write each entity's decodable images to `out_dir/<entity id>/<n><suffix>`.

    Returns ({entity id: file paths relative to out_dir}, archive paths that failed to decode).
    """
    files: dict[int, list[str]] = {}
    unreadable: list[str] = []
    for entity_id in sorted(folder_of_entity):
        kept: list[str] = []
        for info in members[folder_of_entity[entity_id]]:
            payload = archive.read(info)
            if not decodes(payload):
                unreadable.append(_member_name(info))
                continue
            relative = f"{entity_id}/{len(kept)}{PurePosixPath(_member_name(info)).suffix.lower()}"
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            kept.append(relative)
        if kept:
            files[entity_id] = kept
    return files, unreadable
