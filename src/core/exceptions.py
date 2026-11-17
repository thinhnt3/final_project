"""Domain exceptions. Raise where the problem is detected; handle once at the script boundary."""
from __future__ import annotations


class AppError(Exception):
    """Base class for all project errors."""


class ConfigError(AppError):
    """Configuration file is missing a field or holds an invalid value."""


class DataError(AppError):
    """Input data does not match the expected shape or invariants."""


class ChecksumMismatchError(DataError):
    """A raw file on disk differs from the sha256 recorded in the manifest."""


class UpstreamError(AppError):
    """An external service returned an error status or an error payload."""
