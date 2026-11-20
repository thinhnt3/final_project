"""WordNet IsA edges for the image graphs, and indirect queries that need them. No file I/O here.

An object whose concept has a WordNet synset gets its hypernym chain, up to `depth`
levels, as extra concept nodes joined by `is a` relations: horse → equine → odd-toed
ungulate. A hypernym already in the vocabulary reuses its concept id.

An indirect query names a test triplet with its subject replaced by one of the subject's
hypernyms ("equine near man"). Only the IsA edges link the query word to the image's
object, so a model without them must rely on CLIP knowing the category.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.core.exceptions import DataError
from src.data.schemas import Split

ISA_PREDICATE = "is a"
# A synset id → its hypernym synset ids, most specific first (WordNet's first parent each level).
HypernymChain = Callable[[str, int], list[str]]


def extend_vocabulary(
    concepts: Sequence[Mapping[str, Any]],
    predicates: Sequence[Mapping[str, Any]],
    chain: HypernymChain,
    depth: int,
    lemma: Callable[[str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[int]]]:
    """Concepts and predicates with the hypernyms and `is a` added; each concept's chain as ids."""
    if depth < 1:
        raise DataError(f"hypernym depth must be >= 1, got {depth}")
    extended = [dict(c) for c in concepts]
    id_of = {str(c["key"]): int(c["concept_id"]) for c in extended}
    chains: dict[int, list[int]] = {}
    for concept in concepts:
        synset = concept.get("synset")
        if not synset:
            continue
        ids = []
        for parent in chain(str(synset), depth):
            if parent not in id_of:
                id_of[parent] = len(extended)
                extended.append({
                    "concept_id": id_of[parent], "key": parent, "text": lemma(parent), "synset": parent,
                    "train_images": 0, "idf": 0.0,
                })
            ids.append(id_of[parent])
        chains[int(concept["concept_id"])] = ids
    preds = [dict(p) for p in predicates]
    if any(p["text"] == ISA_PREDICATE for p in preds):
        raise DataError(f"predicate {ISA_PREDICATE!r} already exists")
    preds.append({"predicate_id": len(preds), "text": ISA_PREDICATE, "train_images": 0})
    return extended, preds, chains


def add_isa_edges(graph: Mapping[str, Any], chains: Mapping[int, Sequence[int]], isa_id: int) -> dict[str, Any]:
    """Copy of an image graph with every object's hypernym chain as concept nodes and `is a` triplets."""
    concepts = [int(c) for c in graph["concept_ids"]]
    counts = [int(n) for n in graph["concept_counts"]]
    triplets = [[int(x) for x in t] for t in graph["triplets"]]
    present = set(concepts)
    edges: set[tuple[int, int]] = set()
    for concept in list(concepts):
        child = concept
        for parent in chains.get(concept, ()):
            if parent not in present:
                present.add(parent)
                concepts.append(parent)
                counts.append(0)  # not an annotated object
            edges.add((child, parent))
            child = parent
    triplets += [[child, isa_id, parent] for child, parent in sorted(edges)]
    return {**graph, "concept_ids": concepts, "concept_counts": counts, "triplets": triplets}


def build_indirect_queries(
    graphs: Sequence[Mapping[str, Any]],
    chains: Mapping[int, Sequence[int]],
    concept_text: Mapping[int, str],
    predicate_text: Mapping[int, str],
    seed: int,
) -> list[dict[str, Any]]:
    """One indirect query and its direct twin per test image with an eligible triplet.

    `graphs` are the graphs without IsA edges. Eligible: a triplet (s, p, o) with s ≠ o and
    a hypernym h of s such that no other test image has (s', p, o) with s' under h. The
    confusers hold o and some concept under h but no such triplet. The twin names s itself.
    """
    test = sorted((g for g in graphs if g["split"] == Split.TEST.value), key=lambda g: int(g["vg_image_id"]))
    if not test:
        raise DataError("no test image graphs")
    under: dict[int, set[int]] = {}
    for concept, parents in chains.items():
        for parent in parents:
            under.setdefault(parent, set()).add(concept)
    triplets_of = {int(g["vg_image_id"]): {tuple(int(x) for x in t) for t in g["triplets"]} for g in test}
    concepts_of = {int(g["vg_image_id"]): {int(c) for c in g["concept_ids"]} for g in test}
    # (hypernym, predicate, object) → images with a triplet whose subject is under the hypernym.
    general: Counter[tuple[int, int, int]] = Counter()
    for triplets in triplets_of.values():
        general.update({(h, p, o) for s, p, o in triplets for h in chains.get(s, ())})
    rng = random.Random(seed)
    queries: list[dict[str, Any]] = []
    for image_id, triplets in triplets_of.items():
        candidates = []
        for s, p, o in sorted(triplets):
            for h in chains.get(s, ()):
                if s == o or general[(h, p, o)] != 1:
                    continue
                confusers = [
                    i for i, cs in concepts_of.items()
                    if i != image_id and o in cs and cs & under[h]
                    and not any(t[1] == p and t[2] == o and t[0] in under[h] for t in triplets_of[i])
                ]
                if confusers:
                    candidates.append(((s, p, o), h, confusers))
        if not candidates:
            continue
        (s, p, o), h, confusers = rng.choice(candidates)
        for kind, head in (("indirect", h), ("direct", s)):
            queries.append({
                "query_id": f"{kind}-{image_id}",
                "kind": kind,
                "text": f"{concept_text[head]} {predicate_text[p]} {concept_text[o]}",
                "split": Split.TEST.value,
                "relevant_ids": [image_id],
                "confuser_ids": confusers,
                "triplet": [s, p, o],
                "hypernym": h,
            })
    return queries
