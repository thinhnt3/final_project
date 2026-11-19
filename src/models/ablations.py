"""Pure, deterministic graph interventions; source annotations are never mutated."""

from __future__ import annotations

import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Any


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
