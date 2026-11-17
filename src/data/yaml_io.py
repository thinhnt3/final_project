"""YAML write helper for small, human-read summaries of pipeline outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: str | Path, payload: Any, *, header: str = "") -> None:
    """Write YAML atomically, keys in the given order; `header` lines become leading comments."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in header.splitlines():
            handle.write(f"# {line}\n")
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    tmp.replace(target)
