"""Turn the VN heritage core into an entity-retrieval task: text query -> place with photos.

Curated Wikidata claims are the ground truth. A share of them is held out; a held-out claim
(place, relation, target) becomes the query "a place <relation> <target>". Edges read from
Wikipedia articles are never asked about: they stay in the graph as evidence, under their
own relation names, so the experiment measures whether they help recover held-out claims.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

from src.data.mkg_w_config import RelationTemplate
from src.data.mkg_w_queries import Direction, DirectionStat
from src.data.mkg_w_raw import Triple
from src.data.schemas import Split
from src.data.vn_heritage_core import assign_split

NamedTriple = tuple[str, str, str]


def first_label(labels: Mapping[str, str], languages: Sequence[str]) -> str | None:
    """Label in the first available language of `languages`, or None."""
    return next((labels[language] for language in languages if language in labels), None)


def split_claims(
    claims: Iterable[NamedTriple], places: frozenset[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[Split, list[NamedTriple]]:
    """Split the claims of places by the pair of nodes they link; other claims go to train.

    Every claim between the same two nodes lands in one split, so a held-out fact cannot be
    read from its inverse (`part_of` / `has_part`) or from a second property stating it.
    Claims without a place (administrative hierarchy) only give the graph structure.
    """
    by_split: dict[Split, list[NamedTriple]] = {split: [] for split in Split}
    for head, relation, tail in claims:
        if head in places or tail in places:
            first, second = sorted((head, tail))
            split = assign_split(f"{first}|{second}", seed, val_fraction, test_fraction)
        else:
            split = Split.TRAIN
        by_split[split].append((head, relation, tail))
    return by_split


def article_triples(triples: Iterable[NamedTriple], prefix: str) -> list[NamedTriple]:
    """Article edges under prefixed relation names, without duplicates, in first-seen order.

    The prefix keeps "the article says X" apart from the claim X: a cue-based name such as
    `located_in_admin` would otherwise merge with the claim and be removed with it whenever
    that claim is the fact being asked.
    """
    return list(dict.fromkeys((head, f"{prefix}{relation}", tail) for head, relation, tail in triples))


def to_ids(
    triples: Iterable[NamedTriple], entity_id: Mapping[str, int], relation_id: Mapping[str, int]
) -> list[Triple]:
    """Named triples as id triples."""
    return [Triple(entity_id[head], relation_id[relation], entity_id[tail]) for head, relation, tail in triples]


def worded_directions(
    stats: Mapping[tuple[int, Direction], DirectionStat], templates: Mapping[int, RelationTemplate]
) -> dict[tuple[int, Direction], DirectionStat]:
    """Keep the (relation, direction) pairs that have query wording; the others are not asked."""
    return {
        (relation, direction): stat
        for (relation, direction), stat in stats.items()
        if relation in templates and getattr(templates[relation], direction.value) is not None
    }


class DensityTier(StrEnum):
    """How much the graph says around the entity a query names."""

    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"


def triples_within_two_hops(anchors: Iterable[int], triples: Sequence[Triple]) -> dict[int, int]:
    """For each anchor, the number of triples that touch it or one of its neighbours."""
    touching: dict[int, set[int]] = defaultdict(set)
    for row, triple in enumerate(triples):
        touching[triple.head].add(row)
        touching[triple.tail].add(row)
    counts: dict[int, int] = {}
    for anchor in set(anchors):
        rows = set(touching[anchor])
        for row in touching[anchor]:
            rows |= touching[triples[row].head] | touching[triples[row].tail]
        counts[anchor] = len(rows)
    return counts


def density_tier(count: int, sparse_max: int, medium_max: int) -> DensityTier:
    """Tier of a two-hop triple count: up to `sparse_max`, up to `medium_max`, or above."""
    if count <= sparse_max:
        return DensityTier.SPARSE
    return DensityTier.MEDIUM if count <= medium_max else DensityTier.DENSE
