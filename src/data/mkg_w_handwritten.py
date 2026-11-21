"""Check hand-written MKG-W queries against the triples and turn them into query records."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from src.core.exceptions import DataError
from src.data.mkg_w_queries import Direction, QueryGroup
from src.data.mkg_w_raw import Triple
from src.data.schemas import Split

Completions = Mapping[tuple[int, int, Direction], set[int]]


class HandwrittenKind(StrEnum):
    """Fact pattern behind a hand-written query."""

    SINGLE = "single"
    CHAIN = "chain"
    INTERSECTION = "intersection"


def completions_of(triples: Sequence[Triple]) -> dict[tuple[int, int, Direction], set[int]]:
    """(entity, relation, which end is asked) → the entities at that end."""
    found: dict[tuple[int, int, Direction], set[int]] = defaultdict(set)
    for triple in triples:
        found[(triple.head, triple.relation, Direction.TAIL)].add(triple.tail)
        found[(triple.tail, triple.relation, Direction.HEAD)].add(triple.head)
    return found


def pattern_answers(kind: HandwrittenKind, steps: Sequence[tuple[int | None, int, Direction]],
                    completions: Completions) -> set[int]:
    """Entities that satisfy the steps; each step is (anchor or None, relation, asked end)."""
    if kind is HandwrittenKind.CHAIN:
        reached = {anchor for anchor, _, _ in steps[:1] if anchor is not None}
        for _, relation, direction in steps:
            reached = set().union(*(completions.get((entity, relation, direction), set()) for entity in reached))
        return reached
    return set.intersection(*(completions.get((anchor, relation, direction), set())
                              for anchor, relation, direction in steps if anchor is not None))


def resolve_handwritten(
    items: Sequence[Mapping[str, Any]],
    entity_of: Mapping[str, int],
    relation_of: Mapping[str, int],
    triples: Mapping[Split, Sequence[Triple]],
    has_image: Sequence[bool],
) -> list[dict[str, Any]]:
    """Query records for the hand-written items, in file order.

    Raises DataError when a name or relation is unknown, a step is malformed, a declared answer
    does not satisfy the pattern or has no image, or the pattern already holds in the train
    graph (then the query asks for nothing unseen).
    """
    everything = completions_of([t for rows in triples.values() for t in rows])
    train_only = completions_of(triples[Split.TRAIN])
    records = []
    for number, item in enumerate(items, start=1):
        where = f"hand-written query {number} ({item.get('text')!r})"
        try:
            kind = HandwrittenKind(item["kind"])
            steps = [(entity_of[s["anchor"]] if "anchor" in s else None, relation_of[s["relation"]],
                      Direction(s["answer_is"])) for s in item["steps"]]
            declared = [entity_of[name] for name in item["answers"]]
        except (KeyError, ValueError) as error:
            raise DataError(f"{where}: unknown entity, relation, kind or direction {error}") from error
        anchored = [anchor is not None for anchor, _, _ in steps]
        expected = [True] + [False] * (len(steps) - 1) if kind is HandwrittenKind.CHAIN else [True] * len(steps)
        if anchored != expected or (kind is HandwrittenKind.SINGLE and len(steps) != 1) or (
                kind is not HandwrittenKind.SINGLE and len(steps) < 2):
            raise DataError(f"{where}: steps do not fit kind {kind.value}")
        answers = pattern_answers(kind, steps, everything)
        seen_in_train = pattern_answers(kind, steps, train_only)
        for answer in declared:
            if answer not in answers:
                raise DataError(f"{where}: declared answer {answer} does not satisfy the steps")
            if not has_image[answer]:
                raise DataError(f"{where}: declared answer {answer} has no image")
            if answer in seen_in_train:
                raise DataError(f"{where}: answer {answer} already follows from the train graph")
        anchors = [anchor for anchor, _, _ in steps if anchor is not None]
        records.append({
            "query_id": f"hand-{number:02d}",
            "text": item["text"],
            "group": QueryGroup.INDIRECT.value,
            "kind": f"handwritten_{kind.value}",
            "split": Split.TEST.value,
            "relevant_ids": sorted(declared),
            "known_ids": sorted(answers - set(declared) - set(anchors)),
            "anchor_ids": anchors,
            "relation_id": None,
            "direction": None,
            "hops": None,
        })
    return records
