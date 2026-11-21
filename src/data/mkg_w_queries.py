"""Turn MKG-W triples and entity text into retrieval queries whose answers are entities with images.

Two groups. Direct queries name or describe the answer entity. Indirect queries name a
different entity (the anchor) and a relation, so the answer is reachable only through the
graph: "the director of My Darling Clementine".
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.core.exceptions import DataError
from src.data.mkg_w_config import QueryRules, RelationTemplate
from src.data.mkg_w_raw import Triple, display_name, first_sentence
from src.data.schemas import Split

# Chain and hand-written test sets, written by `build_mkg_w_extra_queries` under the processed directory.
EXTRA_QUERIES_FILE = "extra_queries.json"


class Direction(StrEnum):
    """Which end of the triple is the answer."""

    TAIL = "tail"  # query names the head
    HEAD = "head"  # query names the tail


class QueryGroup(StrEnum):
    """Reporting group required by the assignment."""

    DIRECT = "direct"
    INDIRECT = "indirect"


class QueryKind(StrEnum):
    """How the query text was produced."""

    NAME = "name"
    DESCRIPTION = "description"
    RELATION = "relation"
    CHAIN = "chain"  # two facts in a row; the entity between them is not named


class DirectionStatus(StrEnum):
    """Outcome of the query rules for one (relation, direction)."""

    KEPT = "kept"
    TOO_FEW_TRIPLES = "too_few_triples"
    ANSWERS_ARE_CLASSES = "answers_are_classes"


@dataclass(frozen=True)
class EntityQuery:
    """A text query answered by entities.

    `relevant_ids`: correct answers of this split that have images.
    `known_ids`: other correct answers (other splits, or without images); a ranking must
    skip them before scoring, as in filtered link prediction.
    """

    query_id: str
    text: str
    group: QueryGroup
    kind: QueryKind
    split: Split
    relevant_ids: tuple[int, ...]
    known_ids: tuple[int, ...] = ()
    anchor_id: int | None = None
    relation_id: int | None = None
    direction: Direction | None = None
    hops: int | None = None  # shortest train-graph path anchor → answer; None when longer than the cap

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "group": self.group.value,
            "kind": self.kind.value,
            "split": self.split.value,
            "relevant_ids": list(self.relevant_ids),
            "known_ids": list(self.known_ids),
            "anchor_id": self.anchor_id,
            "relation_id": self.relation_id,
            "direction": self.direction.value if self.direction else None,
            "hops": self.hops,
        }


@dataclass(frozen=True)
class DirectionStat:
    """Train statistics of one (relation, direction) and the rule outcome."""

    relation_id: int
    direction: Direction
    train_triples: int
    unique_answers: int
    prior_hits: float  # share of train triples whose answer is among the top-k most frequent answers
    status: DirectionStatus


def _anchor_answer(triple: Triple, direction: Direction) -> tuple[int, int]:
    return (triple.head, triple.tail) if direction is Direction.TAIL else (triple.tail, triple.head)


def direction_stats(train: Sequence[Triple], rules: QueryRules) -> dict[tuple[int, Direction], DirectionStat]:
    """Apply the relation-level rules to every (relation, direction) seen in train."""
    answers: dict[tuple[int, Direction], Counter[int]] = defaultdict(Counter)
    for triple in train:
        for direction in Direction:
            answers[(triple.relation, direction)][_anchor_answer(triple, direction)[1]] += 1
    stats: dict[tuple[int, Direction], DirectionStat] = {}
    for (relation_id, direction), counts in answers.items():
        total = sum(counts.values())
        prior_hits = sum(n for _, n in counts.most_common(rules.prior_top_k)) / total
        if total < rules.min_train_triples:
            status = DirectionStatus.TOO_FEW_TRIPLES
        elif prior_hits > rules.max_prior_hits:
            status = DirectionStatus.ANSWERS_ARE_CLASSES
        else:
            status = DirectionStatus.KEPT
        stats[(relation_id, direction)] = DirectionStat(
            relation_id, direction, total, len(counts), round(prior_hits, 4), status
        )
    return stats


def build_indirect_queries(
    triples: Mapping[Split, Sequence[Triple]],
    stats: Mapping[tuple[int, Direction], DirectionStat],
    templates: Mapping[int, RelationTemplate],
    names: Sequence[str],
    has_image: Sequence[bool],
    max_answers: int,
) -> list[EntityQuery]:
    """One query per (anchor, relation, direction, split) that passes the rules.

    Raises DataError when a kept direction has no template, so wording gaps cannot
    silently remove a relation.
    """
    kept = {key for key, stat in stats.items() if stat.status is DirectionStatus.KEPT}
    missing = sorted(
        (relation_id, direction.value)
        for relation_id, direction in kept
        if relation_id not in templates or getattr(templates[relation_id], direction.value) is None
    )
    if missing:
        raise DataError(f"kept (relation id, direction) pairs without a template: {missing}")

    answers: dict[tuple[int, int, Direction], dict[Split, set[int]]] = defaultdict(lambda: defaultdict(set))
    for split, rows in triples.items():
        for triple in rows:
            for direction in Direction:
                if (triple.relation, direction) in kept:
                    anchor, answer = _anchor_answer(triple, direction)
                    answers[(anchor, triple.relation, direction)][split].add(answer)

    queries: list[EntityQuery] = []
    for (anchor, relation_id, direction), by_split in sorted(answers.items()):
        all_answers = set().union(*by_split.values())
        if len(all_answers) > max_answers:
            continue
        wording = getattr(templates[relation_id], direction.value)
        for split in Split:
            relevant = sorted(a for a in by_split.get(split, ()) if has_image[a])
            if not relevant:
                continue
            queries.append(EntityQuery(
                query_id=f"rel-{anchor}-{relation_id}-{direction.value}-{split.value}",
                text=wording.format(anchor=display_name(names[anchor])),
                group=QueryGroup.INDIRECT,
                kind=QueryKind.RELATION,
                split=split,
                relevant_ids=tuple(relevant),
                known_ids=tuple(sorted(all_answers - set(relevant))),
                anchor_id=anchor,
                relation_id=relation_id,
                direction=direction,
            ))
    return queries


def build_chain_queries(
    triples: Mapping[Split, Sequence[Triple]],
    split: Split,
    stats: Mapping[tuple[int, Direction], DirectionStat],
    templates: Mapping[int, RelationTemplate],
    names: Sequence[str],
    has_image: Sequence[bool],
    max_answers: int,
) -> list[EntityQuery]:
    """Queries that need two facts in a row: "the performer of the work that came right after X".

    The first fact (anchor → middle entity) is a train triple, so it is in the graph. The second
    (middle → answer) is a triple of `split`, so it is not. The middle entity is never named in
    the text, and the first step must have exactly one completion over all triples, so the
    inner phrase denotes one entity. Same relation rules, wording and answer cap as the
    single-fact queries.
    """
    kept = {key for key, stat in stats.items() if stat.status is DirectionStatus.KEPT}
    completions: dict[tuple[int, int, Direction], set[int]] = defaultdict(set)
    for rows in triples.values():
        for triple in rows:
            for direction in Direction:
                anchor, answer = _anchor_answer(triple, direction)
                completions[(anchor, triple.relation, direction)].add(answer)
    # Middle entity → the (anchor, relation, direction) train steps that lead to it and nowhere else.
    steps_into: dict[int, list[tuple[int, int, Direction]]] = defaultdict(list)
    for triple in triples[Split.TRAIN]:
        for direction in Direction:
            anchor, middle = _anchor_answer(triple, direction)
            key = (anchor, triple.relation, direction)
            if (triple.relation, direction) in kept and len(completions[key]) == 1:
                steps_into[middle].append(key)

    split_answers: dict[tuple[int, int, Direction], set[int]] = defaultdict(set)
    for triple in triples[split]:
        for direction in Direction:
            if (triple.relation, direction) in kept:
                middle, answer = _anchor_answer(triple, direction)
                split_answers[(middle, triple.relation, direction)].add(answer)

    queries: list[EntityQuery] = []
    for (middle, relation_id, direction), answers in sorted(split_answers.items()):
        all_answers = completions[(middle, relation_id, direction)]
        if len(all_answers) > max_answers:
            continue
        outer = getattr(templates[relation_id], direction.value)
        for anchor, first_relation, first_direction in sorted(steps_into[middle]):
            relevant = sorted(a for a in answers if has_image[a] and a != anchor)
            if not relevant:
                continue
            inner = getattr(templates[first_relation], first_direction.value)
            queries.append(EntityQuery(
                query_id=(f"chain-{anchor}-{first_relation}-{first_direction.value}"
                          f"-{relation_id}-{direction.value}-{split.value}"),
                text=outer.format(anchor=inner.format(anchor=display_name(names[anchor]))),
                group=QueryGroup.INDIRECT,
                kind=QueryKind.CHAIN,
                split=split,
                relevant_ids=tuple(relevant),
                known_ids=tuple(sorted(all_answers - set(relevant) - {anchor})),
                anchor_id=anchor,
            ))
    return queries


def anchor_answer_hops(queries: Sequence[EntityQuery], train: Sequence[Triple], max_hops: int) -> list[int | None]:
    """For each query, the fewest train edges (either direction) from its anchor to a relevant answer.

    Val and test facts are not in the train graph, so any path found there is evidence the
    model can reach without seeing the answer. A train query's own fact is in the graph; the
    direct edge counts as one hop only when a different relation also links the pair.
    None for queries without an anchor and when no path has at most `max_hops` edges.
    """
    neighbors: dict[int, set[int]] = defaultdict(set)
    pair_relations: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triple in train:
        neighbors[triple.head].add(triple.tail)
        neighbors[triple.tail].add(triple.head)
        pair_relations[(min(triple.head, triple.tail), max(triple.head, triple.tail))].add(triple.relation)

    def hops_of(query: EntityQuery) -> int | None:
        anchor = query.anchor_id
        if anchor is None:
            return None
        targets = set(query.relevant_ids)
        own_edge_only = {
            answer for answer in targets
            if query.split is Split.TRAIN
            and pair_relations[(min(anchor, answer), max(anchor, answer))] <= {query.relation_id}
        }
        seen, frontier = {anchor}, {anchor}
        for hops in range(1, max_hops + 1):
            reached: set[int] = set()
            for node in frontier:
                for neighbor in neighbors[node]:
                    if hops == 1 and neighbor in own_edge_only:
                        continue  # reachable here only through the fact being asked
                    if neighbor in targets:
                        return hops
                    if neighbor not in seen:
                        seen.add(neighbor)
                        reached.add(neighbor)
            frontier = reached
        return None

    return [hops_of(query) for query in queries]


def split_entities(entity_ids: Sequence[int], fractions: Mapping[Split, float], seed: int) -> dict[Split, list[int]]:
    """Seeded random partition of the entity ids by `fractions`; test takes the remainder."""
    shuffled = sorted(entity_ids)
    random.Random(seed).shuffle(shuffled)
    n_train = round(len(shuffled) * fractions[Split.TRAIN])
    n_val = round(len(shuffled) * fractions[Split.VAL])
    return {
        Split.TRAIN: sorted(shuffled[:n_train]),
        Split.VAL: sorted(shuffled[n_train:n_train + n_val]),
        Split.TEST: sorted(shuffled[n_train + n_val:]),
    }


def build_direct_queries(
    entity_split: Mapping[Split, Sequence[int]],
    names: Sequence[str],
    descriptions: Mapping[int, str],
) -> list[EntityQuery]:
    """A name query per entity, plus a first-sentence description query when it has one."""
    queries: list[EntityQuery] = []
    for split, entity_ids in entity_split.items():
        for entity_id in entity_ids:
            texts = [(QueryKind.NAME, display_name(names[entity_id]))]
            if entity_id in descriptions:
                texts.append((QueryKind.DESCRIPTION, first_sentence(descriptions[entity_id])))
            queries.extend(
                EntityQuery(
                    query_id=f"{kind.value}-{entity_id}",
                    text=text,
                    group=QueryGroup.DIRECT,
                    kind=kind,
                    split=split,
                    relevant_ids=(entity_id,),
                )
                for kind, text in texts if text
            )
    return queries
