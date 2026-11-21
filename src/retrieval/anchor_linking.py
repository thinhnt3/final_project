"""Find the entities a query names by matching entity names in its text."""
from __future__ import annotations

import re
from collections.abc import Sequence

from src.core.exceptions import DataError

_WORD = re.compile(r"\S+")
# Punctuation that can stick to a name inside a sentence: "Doggfather?", "(Doggfather)", "Doggfather's".
_EDGE_PUNCTUATION = "\"'“”‘’.,;:!?"
_POSSESSIVE = re.compile(r"['’]s$")


def _variants(span: str) -> list[str]:
    """The span as written, then without sentence punctuation or a possessive ending."""
    stripped = span.strip(_EDGE_PUNCTUATION)
    return list(dict.fromkeys([span, stripped, _POSSESSIVE.sub("", stripped)]))


class AnchorLinker:
    """Exact, case-sensitive match of entity names against the words of a query.

    Entity names are proper nouns written with capitals ("Film", "London"), so matching case
    keeps the common words of a question ("a film written by …") from becoming anchors.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self._entity_of: dict[str, int] = {}
        for entity_id, name in enumerate(names):
            if not name:
                raise DataError(f"entity {entity_id} has no name")
            if name in self._entity_of:
                raise DataError(f"entities {self._entity_of[name]} and {entity_id} share the name {name!r}")
            self._entity_of[name] = entity_id
        self._max_words = max(len(name.split()) for name in names)

    def link(self, text: str) -> list[int]:
        """Entity ids named in `text`, in order of appearance; the longest name wins where names overlap."""
        words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        found: list[int] = []
        position = 0
        while position < len(words):
            for length in range(min(self._max_words, len(words) - position), 0, -1):
                span = text[words[position][0]:words[position + length - 1][1]]
                entity = next((self._entity_of[v] for v in _variants(span) if v in self._entity_of), None)
                if entity is not None:
                    found.append(entity)
                    position += length
                    break
            else:
                position += 1
        return list(dict.fromkeys(found))
