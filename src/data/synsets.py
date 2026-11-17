"""Pure text helpers that turn Visual Genome labels into canonical concept labels."""
from __future__ import annotations

import re
import string

from src.core.exceptions import DataError

_WHITESPACE = re.compile(r"\s+")
_EDGE_CHARS = string.punctuation + string.whitespace


def normalize_text(text: str) -> str:
    """Lowercase, collapse inner whitespace, strip punctuation at both ends."""
    return _WHITESPACE.sub(" ", text.lower()).strip(_EDGE_CHARS)


_TRAILING_ARTICLES = frozenset({"a", "an", "the"})
_LEADING_AUXILIARIES = frozenset({"is", "are"})


def normalize_predicate(text: str) -> str:
    """Normalize a VG predicate: `normalize_text`, then drop trailing articles and a leading
    `is`/`are` (`are on a` → `on`, `has a` → `has`). Inflections are kept (`wears` ≠ `wearing`)."""
    tokens = normalize_text(text).split()
    while tokens and tokens[-1] in _TRAILING_ARTICLES:
        tokens.pop()
    while tokens and tokens[0] in _LEADING_AUXILIARIES:
        tokens.pop(0)
    return " ".join(tokens)


def synset_lemma(synset: str) -> str:
    """WordNet synset id → readable lemma: `gym_shoe.n.01` → `gym shoe`."""
    parts = synset.rsplit(".", 2)
    if len(parts) != 3 or not parts[0] or not parts[2].isdigit():
        raise DataError(f"malformed synset id: {synset!r}")
    return parts[0].replace("_", " ").lower()
