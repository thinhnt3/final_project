"""Pure transforms that cut a dense core out of the VN heritage graph and add admin hierarchy.

The full build keeps every place that has a photo, so half of the places carry no knowledge
besides their type. The core keeps the well-described places and links their administrative
units to each other, which gives the graph paths longer than place -> hub -> place.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.data.schemas import Split

Triple = tuple[str, str, str]
_CENTURY_PRECISION = 7  # Wikidata time precision: 7 = century, 9 = year


@dataclass(frozen=True)
class CoreRules:
    """Thresholds and relation roles used to select the core; see the YAML comments."""

    min_triples: int
    min_images: int
    label_language: str
    class_relations: frozenset[str]
    shadowed_relations: dict[str, str]
    admin_relations: frozenset[str]
    hierarchy_relation: str
    group_relations: frozenset[str]
    province_classes: frozenset[str]


def _claim_values(entity: dict[str, Any], prop: str) -> list[Any]:
    """Values of the non-deprecated claims of one property."""
    return [
        claim["mainsnak"]["datavalue"]["value"]
        for claim in entity.get("claims", {}).get(prop, [])
        if claim.get("rank") != "deprecated" and claim["mainsnak"].get("snaktype") == "value"
    ]


def claim_item_ids(entity: dict[str, Any], prop: str) -> list[str]:
    """Item ids one entity points to through a property (deprecated claims skipped)."""
    return [value["id"] for value in _claim_values(entity, prop) if isinstance(value, dict) and "id" in value]


def item_triples(entities: dict[str, dict[str, Any]], relations: dict[str, str]) -> list[Triple]:
    """One triple per item-valued claim of the configured properties, for every given entity."""
    triples: dict[Triple, None] = {}
    for qid in sorted(entities):
        for prop, relation in relations.items():
            for target in claim_item_ids(entities[qid], prop):
                triples[(qid, relation, target)] = None
    return list(triples)


def inception_century(entity: dict[str, Any], prop: str) -> int | None:
    """Century of the first usable date claim, or None.

    Dates before year 1 and dates known only to the millennium are skipped; a century-precision
    value such as 1900 denotes the century that ends in that year.
    """
    for value in _claim_values(entity, prop):
        year = int(value["time"][:5]) if value["time"][0] in "+-" else int(value["time"][:4])
        if year > 0 and value.get("precision", 0) >= _CENTURY_PRECISION:
            return (year - 1) // 100 + 1
    return None


def century_node(century: int) -> tuple[str, dict[str, str]]:
    """Synthetic node id and labels for a century; Wikidata has no claim pointing at one."""
    return f"century:{century}", {"vi": f"thế kỷ {century}", "en": f"century {century}"}


def class_like_places(place_qids: set[str], triples: Iterable[Triple], class_relations: frozenset[str]) -> set[str]:
    """Places that other places point to as a class or designation (e.g. "chùa Việt Nam").

    They are categories admitted by the title-prefix rule, not sites that can be photographed.
    """
    return {tail for _, relation, tail in triples if relation in class_relations and tail in place_qids}


def drop_shadowed(triples: list[Triple], shadowed_relations: dict[str, str]) -> list[Triple]:
    """Remove a triple when the same (head, tail) pair already exists under the shadowing relation."""
    present = set(triples)
    return [
        (head, relation, tail)
        for head, relation, tail in triples
        if (head, shadowed_relations.get(relation, ""), tail) not in present
    ]


def select_core(
    labels: dict[str, dict[str, str]], triples: list[Triple], image_counts: dict[str, int], rules: CoreRules
) -> list[str]:
    """Place ids with enough knowledge triples, enough images and a label in the target language."""
    excluded = class_like_places(set(labels), triples, rules.class_relations)
    degree = Counter(head for head, _, _ in triples)
    return sorted(
        qid
        for qid, place_labels in labels.items()
        if qid not in excluded
        and rules.label_language in place_labels
        and degree[qid] >= rules.min_triples
        and image_counts.get(qid, 0) >= rules.min_images
    )


def admin_hierarchy(
    start: Iterable[str],
    parents: dict[str, list[str]],
    classes: dict[str, list[str]],
    province_classes: frozenset[str],
    labelled: set[str],
    relation: str,
) -> tuple[list[Triple], int]:
    """Edges unit -> parent unit, walking P131 upwards from `start` and stopping at provinces.

    Returns the edges and the number of parents skipped because they have no label; an
    unlabelled node cannot be shown in an explanation, so its edge is left out.
    """
    edges: set[Triple] = set()
    skipped: set[str] = set()
    frontier, seen = set(start), set()
    while frontier:
        seen |= frontier
        reached: set[str] = set()
        for unit in frontier:
            if province_classes.intersection(classes.get(unit, [])):
                continue
            for parent in parents.get(unit, []):
                if parent not in labelled:
                    skipped.add(parent)
                    continue
                edges.add((unit, relation, parent))
                reached.add(parent)
        frontier = reached - seen
    return sorted(edges), len(skipped)


def split_groups(
    place_qids: Iterable[str], triples: Iterable[Triple], group_relations: frozenset[str]
) -> dict[str, str]:
    """Map each place to the smallest id of its group; places joined by a group relation share one.

    Two towers of one sanctuary must not sit in different splits, whether they point at each
    other or at a shared third node.
    """
    root: dict[str, str] = {}

    def find(node: str) -> str:
        root.setdefault(node, node)
        while root[node] != node:
            root[node] = root[root[node]]
            node = root[node]
        return node

    for head, relation, tail in triples:
        if relation in group_relations:
            low, high = sorted((find(head), find(tail)))
            root[high] = low
    members: dict[str, list[str]] = defaultdict(list)
    for qid in place_qids:
        members[find(qid)].append(qid)
    return {qid: min(group) for group in members.values() for qid in group}


def assign_split(key: str, seed: int, val_fraction: float, test_fraction: float) -> Split:
    """Deterministic split from a hash of (seed, key); the same rule the full build uses per place."""
    bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return Split.TEST
    if bucket < test_fraction + val_fraction:
        return Split.VAL
    return Split.TRAIN
