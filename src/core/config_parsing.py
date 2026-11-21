"""Helpers for reading typed fields out of a parsed YAML mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigError


def read_mapping(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file whose top level is a mapping; raise ConfigError otherwise."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")
    return raw


def require_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return the nested mapping under `key`; raise ConfigError if it is not a mapping."""
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def require_field(raw: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    """Return `raw[key]` if it has the given type; raise ConfigError otherwise."""
    value = raw.get(key)
    # bool is a subclass of int; reject it so `workers: true` fails, unless a bool is asked for.
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ConfigError(f"field '{key}' must be {kind}, got {value!r}")
    return value
