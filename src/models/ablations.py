"""Pure relation perturbations for the existing image/object/relation graph schema."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.core.exceptions import ConfigError, DataError
from src.models.graph_data import image_graph_to_data


def remove_relations(image_graph: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    """Remove relation instances by position, keeping every image and object feature."""
    image_graph_to_data(image_graph, 0)
    triplets = image_graph["triplets"]
    if any(type(i) is not int or not 0 <= i < len(triplets) for i in indices):
        raise DataError("relation index outside the graph")
    removed = set(indices)
    result = deepcopy(dict(image_graph))
    result["triplets"] = [list(t) for i, t in enumerate(triplets) if i not in removed]
    return result


def objects_only(image_graph: Mapping[str, Any]) -> dict[str, Any]:
    """A2: drop all relation nodes; retain image → object edges when rebuilt."""
    return remove_relations(image_graph, list(range(len(image_graph["triplets"]))))


@dataclass(frozen=True)
class RelationRewiring:
    """Rewired graph plus evidence of how much randomization was possible."""

    graph: dict[str, Any]
    requested_swaps: int
    successful_swaps: int
    attempts: int
    changed_relations: int


def rewire_relations(image_graph: Mapping[str, Any], *, seed: int, swaps: int, max_attempts: int) -> RelationRewiring:
    """A4: swap relation targets, preserving directed object degrees and predicate counts.

    (s, p, o), (u, q, v) become (s, p, v), (u, q, o). New self loops or duplicate
    triplets are rejected. Existing duplicates/self loops are left intact. Small graphs
    can have no valid swap; callers must report the counters rather than claim success.
    Seed depends on image id so batching or evaluation order cannot affect the variant.
    """
    if any(type(value) is not int or value < 0 for value in (seed, swaps, max_attempts)):
        raise ConfigError("seed, swaps and max_attempts must be nonnegative integers")
    image_graph_to_data(image_graph, 0)
    original = [tuple(int(v) for v in t) for t in image_graph["triplets"]]
    triplets = original.copy()
    counts = Counter(triplets)
    rng = random.Random(f"{seed}:{image_graph['vg_image_id']}")
    successful = attempts = 0
    while len(triplets) >= 2 and successful < swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(len(triplets)), 2)
        first, second = triplets[i], triplets[j]
        s, p, o = first
        u, q, v = second
        if s in (u, v) or o == v or u == o:
            continue
        left, right = (s, p, v), (u, q, o)
        if counts[left] or counts[right]:
            continue
        counts[first] -= 1
        counts[second] -= 1
        counts[left] += 1
        counts[right] += 1
        triplets[i], triplets[j] = left, right
        successful += 1
    result = deepcopy(dict(image_graph))
    result["triplets"] = [list(t) for t in triplets]
    return RelationRewiring(
        result, swaps, successful, attempts, sum(a != b for a, b in zip(original, triplets, strict=True))
    )
