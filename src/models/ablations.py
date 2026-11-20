"""Pure, deterministic graph interventions; source annotations are never mutated."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.core.exceptions import DataError


class Ablation(StrEnum):
    FULL = "full"
    OBJECTS_ONLY = "objects_only"
    DROP_PREDICATES = "drop_predicates"
    RANDOM_EDGES = "random_edges"
    UNDIRECTED = "undirected"


def ablate_graph(
    graph: Mapping[str, Any],
    variant: Ablation,
    seed: int = 0,
) -> dict[str, Any]:
    """Copy a raw image graph, optionally deleting or rewiring its relations.

    Rewiring independently permutes subject and object endpoints. This preserves the
    directed degree of every concept and predicate counts, but permits self-relations
    and duplicate triples (a directed configuration-model control, not a simple graph).
    A private RNG keyed by image id makes results independent of input order and splits.
    Predicate masking and direction removal happen in the encoder, not here.
    """
    # Copy mutable annotation fields before applying an intervention.
    result = dict(graph)
    result["concept_ids"] = list(graph["concept_ids"])
    triples = [list(t) for t in graph["triplets"]]
    if variant is Ablation.OBJECTS_ONLY:
        triples = []
    elif variant is Ablation.RANDOM_EDGES:
        # Shuffle endpoints with a private per-image RNG, preserving degrees.
        rng = random.Random(f"{seed}:{int(graph['vg_image_id'])}")
        subjects = [t[0] for t in triples]
        objects = [t[2] for t in triples]
        rng.shuffle(subjects)
        rng.shuffle(objects)
        triples = [[s, t[1], o] for s, t, o in zip(subjects, triples, objects, strict=True)]
    result["triplets"] = triples
    return result


def remove_relations(
    graph: Mapping[str, Any],
    positions: Sequence[int],
) -> dict[str, Any]:
    """Remove entire triplets by position, preserving all concept nodes."""
    # Remove relation positions only; object nodes stay available.
    indices = set(positions)
    if len(indices) != len(positions) or any(i < 0 or i >= len(graph["triplets"]) for i in indices):
        raise DataError("relation indices must be distinct and in range")
    return {**graph, "triplets": [t for i, t in enumerate(graph["triplets"]) if i not in indices]}


def remove_triplet(graph: Mapping[str, Any], triplet: Sequence[int]) -> dict[str, Any]:
    """Remove one (subject, predicate, object) triplet given by value; graphs hold each triplet once."""
    target = [int(x) for x in triplet]
    for position, candidate in enumerate(graph["triplets"]):
        if [int(x) for x in candidate] == target:
            return remove_relations(graph, [position])
    raise DataError(f"image {graph['vg_image_id']} has no triplet {target}")


@dataclass(frozen=True)
class Corruption:
    """Probabilities of the three kinds of scene-graph noise."""

    drop_edge: float = 0.0
    drop_node: float = 0.0
    flip_label: float = 0.0

    def name(self) -> str:
        """Short label, e.g. `clean` or `drop_edge=0.3`."""
        parts = [f"{k}={v:g}" for k, v in vars(self).items() if v > 0]
        return ",".join(parts) or "clean"


def corrupt_graph(graph: Mapping[str, Any], noise: Corruption, n_concepts: int, seed: int) -> dict[str, Any]:
    """A corrupted copy of one image graph; the same seed, image and setting always give the same copy.

    Dropping an object also drops its relations. A flipped label becomes a random concept
    that is not already in the image, so the graph keeps one node per concept.
    """
    rng = random.Random(f"{seed}-{graph['vg_image_id']}-{noise.name()}")
    concepts = [int(c) for c in graph["concept_ids"]]
    triplets = [tuple(int(x) for x in t) for t in graph["triplets"]]
    kept = [c for c in concepts if rng.random() >= noise.drop_node]
    triplets = [t for t in triplets if t[0] in kept and t[2] in kept]
    triplets = [t for t in triplets if rng.random() >= noise.drop_edge]
    if noise.flip_label > 0:
        present, mapping = set(kept), {}
        for concept in kept:
            if rng.random() < noise.flip_label:
                replacement = rng.randrange(n_concepts)
                while replacement in present:
                    replacement = rng.randrange(n_concepts)
                present.add(replacement)
                mapping[concept] = replacement
        kept = [mapping.get(c, c) for c in kept]
        triplets = [(mapping.get(s, s), p, mapping.get(o, o)) for s, p, o in triplets]
    return {**graph, "concept_ids": kept, "triplets": [list(t) for t in triplets]}
