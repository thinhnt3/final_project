"""Relation queries: test queries that only a model reading the scene-graph relations can solve.

Each query comes from one test triplet (subject, predicate, object) that occurs in exactly
one test image, so it has one relevant image. Its confusers are the other test images that
contain both the subject and the object concept but not that triplet: telling them apart
from the relevant image needs the relation, not the objects.

Every relation query has an objects-only twin on the same image, built from the same two
concepts without the predicate. If a model gains as much on the twin as on the relation
query, it does not use the relation. No file I/O here.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from src.core.exceptions import DataError
from src.data.schemas import Split


class QueryKind(StrEnum):
    """What the query text names: the triplet, or only its two objects."""

    RELATION = "relation"
    OBJECTS = "objects"


TEMPLATES = {
    QueryKind.RELATION: "{subject} {predicate} {object}",
    QueryKind.OBJECTS: "{subject} and {object}",
}

Triplet = tuple[int, int, int]


def _triplets(graph: Mapping[str, Any]) -> set[Triplet]:
    return {(int(s), int(p), int(o)) for s, p, o in graph["triplets"]}


def build_relation_queries(
    graphs: Sequence[Mapping[str, Any]],
    concept_text: Mapping[int, str],
    predicate_text: Mapping[int, str],
    seed: int,
    split: Split = Split.TEST,
) -> list[dict[str, Any]]:
    """One relation query and its objects twin per image of `split` that has an eligible triplet.

    Eligible: subject and object are different concepts, the triplet occurs in no other image
    of the split, and at least one other image of the split holds both concepts. The triplet of an image is
    drawn with `seed`, so the same inputs always give the same queries.
    """
    test = sorted((g for g in graphs if g["split"] == split.value), key=lambda g: int(g["vg_image_id"]))
    if not test:
        raise DataError(f"no {split.value} image graphs")
    occurrences: Counter[Triplet] = Counter(t for g in test for t in _triplets(g))
    concepts_of = {int(g["vg_image_id"]): {int(c) for c in g["concept_ids"]} for g in test}
    rng = random.Random(seed)
    queries: list[dict[str, Any]] = []
    for graph in test:
        image_id = int(graph["vg_image_id"])
        candidates = []
        for triplet in sorted(_triplets(graph)):
            subject, _, obj = triplet
            if subject == obj or occurrences[triplet] != 1:
                continue
            confusers = [i for i, c in concepts_of.items() if i != image_id and subject in c and obj in c]
            if confusers:
                candidates.append((triplet, confusers))
        if not candidates:
            continue
        triplet, confusers = rng.choice(candidates)
        words = {
            "subject": concept_text[triplet[0]],
            "predicate": predicate_text[triplet[1]],
            "object": concept_text[triplet[2]],
        }
        for kind in QueryKind:
            queries.append({
                "query_id": f"{kind.value}-{image_id}",
                "kind": kind.value,
                "text": TEMPLATES[kind].format(**words),
                "split": split.value,
                "relevant_ids": [image_id],
                "confuser_ids": confusers,
                "triplet": list(triplet),
            })
    return queries


def confuser_ranks(
    scores: np.ndarray, relevant_columns: Sequence[int], confuser_columns: Sequence[Sequence[int]]
) -> np.ndarray:
    """1-based rank of the relevant image among itself and its confusers only.

    Ties count against the relevant image, as in `ranks_of_relevant`.
    """
    if len(relevant_columns) != scores.shape[0] or len(confuser_columns) != scores.shape[0]:
        raise DataError("one relevant column and one confuser list are needed per query")
    ranks = []
    for row, (relevant, confusers) in enumerate(zip(relevant_columns, confuser_columns, strict=True)):
        if not confusers:
            raise DataError(f"query row {row} has no confusers")
        ranks.append(1 + int((scores[row, list(confusers)] >= scores[row, relevant]).sum()))
    return np.asarray(ranks)


def chance_accuracy(confuser_columns: Sequence[Sequence[int]]) -> float:
    """Expected top-1 accuracy among confusers of a random ranking."""
    return float(np.mean([1.0 / (1 + len(c)) for c in confuser_columns]))


def contrast_confusers(query: Mapping[str, Any], graph_of: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """Confusers of a relation query that link its two objects by another relation, in either direction.

    Against these, the graph differs from the relevant image only in which relation joins the
    pair; the other confusers have no annotated relation between the pair at all.
    """
    subject, _, obj = (int(x) for x in query["triplet"])
    pair = {subject, obj}
    return [
        image_id for image_id in query["confuser_ids"]
        if any({int(s), int(o)} == pair for s, _, o in graph_of[int(image_id)]["triplets"])
    ]
