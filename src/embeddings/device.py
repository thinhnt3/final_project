"""Compute device choices for model inference."""
from __future__ import annotations

from enum import StrEnum


class Device(StrEnum):
    """Where a model runs; `auto` picks the best available."""

    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
