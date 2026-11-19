"""Does a caption name a phrase? Whole words, lower case, an optional plural `s`/`es`."""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache


@lru_cache(maxsize=8192)
def _pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in phrase.lower().split()]
    if not words:
        raise ValueError("empty phrase")
    words[-1] += "(?:s|es)?"
    return re.compile(r"\b" + r"\s+".join(words) + r"\b")


def names(text: str, phrase: str) -> bool:
    """True if `text` contains `phrase` as whole words, the last one possibly plural."""
    return _pattern(phrase).search(text.lower()) is not None


def names_any(text: str, phrases: Iterable[str]) -> bool:
    """True if `text` names at least one of `phrases`."""
    return any(names(text, p) for p in phrases)
