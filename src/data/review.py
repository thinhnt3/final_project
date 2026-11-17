"""Pure statistics for the data review (Phase 2, step 2.6). No file I/O here."""
from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Function words ignored when matching caption words to graph labels.
STOPWORDS = frozenset("""
a an the and or of on in at to for with by from into onto over under near next up down out off
is are was were be been being has have had there here this that these those it its their his her
some two three four five several many one other another very while as top front side back
""".split())

_TOKEN = re.compile(r"[a-z]+")


def content_tokens(text: str) -> list[str]:
    """Lowercase alphabetic tokens of a caption, minus stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def graph_vocabulary(graph: dict[str, Any]) -> set[str]:
    """Single words from object labels and raw names (raw names keep plural forms)."""
    words: set[str] = set()
    for obj in graph["objects"]:
        words.update(_TOKEN.findall(obj["label"]))
        words.update(_TOKEN.findall(obj["name_raw"]))
    return words


def token_matches(token: str, vocabulary: set[str]) -> bool:
    """Exact match, or match after dropping a plural `s` on either side."""
    return token in vocabulary or token.rstrip("s") in vocabulary or f"{token}s" in vocabulary


def caption_coverage(caption: str, vocabulary: set[str]) -> float | None:
    """Share of the caption's content words found in the vocabulary; None if no content words."""
    tokens = content_tokens(caption)
    if not tokens:
        return None
    return sum(token_matches(t, vocabulary) for t in tokens) / len(tokens)


@dataclass(frozen=True)
class CoverageStats:
    """Caption–graph match for the true image versus a random other image."""

    captions: int
    own_mean: float
    random_mean: float
    own_any_match_pct: float
    random_any_match_pct: float


def coverage_vs_random(
    captions_by_image: dict[int, Sequence[str]], vocab_by_image: dict[int, set[str]], seed: int
) -> CoverageStats:
    """Compare each caption against its own graph and against one random other graph."""
    rng = random.Random(seed)
    image_ids = sorted(captions_by_image)
    own, other = [], []
    for image_id in image_ids:
        for caption in captions_by_image[image_id]:
            decoy = image_id
            while decoy == image_id:
                decoy = rng.choice(image_ids)
            own_cov = caption_coverage(caption, vocab_by_image[image_id])
            if own_cov is None:
                continue
            own.append(own_cov)
            other.append(caption_coverage(caption, vocab_by_image[decoy]) or 0.0)
    return CoverageStats(
        captions=len(own),
        own_mean=_round(statistics.fmean(own)),
        random_mean=_round(statistics.fmean(other)),
        own_any_match_pct=_round(100 * sum(c > 0 for c in own) / len(own)),
        random_any_match_pct=_round(100 * sum(c > 0 for c in other) / len(other)),
    )


def describe(values: Sequence[float]) -> dict[str, float]:
    """mean / p10 / median / p90 / max of a numeric sample."""
    ordered = sorted(values)
    deciles = statistics.quantiles(ordered, n=10)
    return {
        "mean": _round(statistics.fmean(ordered)),
        "p10": _round(deciles[0]),
        "median": _round(statistics.median(ordered)),
        "p90": _round(deciles[-1]),
        "max": _round(ordered[-1]),
    }


def document_frequency(items_per_image: Iterable[Iterable[str]]) -> Counter[str]:
    """Number of images each item appears in (counted once per image)."""
    counts: Counter[str] = Counter()
    for items in items_per_image:
        counts.update(set(items))
    return counts


def _round(value: float) -> float:
    return round(value, 4)
