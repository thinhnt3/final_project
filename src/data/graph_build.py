"""Fixed graph vocabulary and per-image graphs over it. No file I/O here.

The vocabulary (concepts, predicates, display text, IDF) is computed from train images only,
so validation and test images never influence it. Every image, in any split, is then mapped
onto that vocabulary; objects and relations outside it are dropped and counted.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.core.exceptions import DataError
from src.data.review import describe
from src.data.schemas import Concept, ImageGraph, Predicate, SceneGraph, SceneObject, Split

# Key prefix for concepts whose objects have no synset; synset ids never contain a colon.
NAME_KEY_PREFIX = "name:"
_IDF_DECIMALS = 6
_TOP_HUBS = 20


class GraphDropReason(StrEnum):
    """Why an object or relation did not make it into an image graph."""

    OBJECT_JUNK_NAME = "object_junk_name"
    OBJECT_JUNK_CONCEPT = "object_junk_concept"
    OBJECT_OUT_OF_VOCABULARY = "object_out_of_vocabulary"
    RELATION_ENDPOINT_DROPPED = "relation_endpoint_dropped"
    RELATION_PREDICATE_OUT_OF_VOCABULARY = "relation_predicate_out_of_vocabulary"
    RELATION_DUPLICATE = "relation_duplicate"


OBJECT_DROP_REASONS = (
    GraphDropReason.OBJECT_JUNK_NAME,
    GraphDropReason.OBJECT_JUNK_CONCEPT,
    GraphDropReason.OBJECT_OUT_OF_VOCABULARY,
)


def concept_key(obj: SceneObject) -> str:
    """Identity of the concept an object belongs to: its synset, else its raw name."""
    return obj.synset if obj.synset is not None else f"{NAME_KEY_PREFIX}{obj.name_raw}"


@dataclass(frozen=True)
class GraphVocabulary:
    """Concepts and predicates kept in the graph, ordered by id."""

    concepts: tuple[Concept, ...]
    predicates: tuple[Predicate, ...]
    train_images: int
    # Concepts whose usual name is a junk name ("side", "center"). All their objects are
    # dropped, so a leftover variant such as "right side" never becomes a display text.
    junk_concept_keys: frozenset[str]

    def concept_ids(self) -> dict[str, int]:
        """Concept key → id."""
        return {c.key: c.concept_id for c in self.concepts}

    def predicate_ids(self) -> dict[str, int]:
        """Predicate text → id."""
        return {p.text: p.predicate_id for p in self.predicates}


def build_vocabulary(
    train_graphs: Sequence[SceneGraph], *, min_train_images: int, junk_names: frozenset[str]
) -> GraphVocabulary:
    """Keep concepts and predicates seen in at least `min_train_images` train images.

    A concept whose most common raw name is a junk name is junk as a whole. A predicate is
    counted only on relations whose two endpoints are kept concepts, because only those
    relations can become edges. Ids follow descending image count, then key.
    """
    if not train_graphs:
        raise DataError("cannot build a graph vocabulary from zero train images")

    name_images: dict[str, Counter[str]] = {}  # concept key → raw name → train images using it
    for graph in train_graphs:
        for key, name in {(concept_key(obj), obj.name_raw) for obj in graph.objects}:
            name_images.setdefault(key, Counter())[name] += 1
    junk_keys = frozenset(
        key for key, counts in name_images.items() if _most_common_name(counts) in junk_names
    )

    concept_images: Counter[str] = Counter()
    for graph in train_graphs:
        concept_images.update({
            concept_key(obj) for obj in graph.objects if not _is_junk(obj, junk_names, junk_keys)
        })
    kept_keys = {key for key, n in concept_images.items() if n >= min_train_images}
    n_train = len(train_graphs)
    concepts = tuple(
        Concept(
            concept_id=index,
            key=key,
            # Never a junk name: a concept whose most common name is junk was dropped above.
            text=_most_common_name(name_images[key]),
            synset=None if key.startswith(NAME_KEY_PREFIX) else key,
            train_images=concept_images[key],
            idf=round(math.log(n_train / concept_images[key]), _IDF_DECIMALS),
        )
        for index, key in enumerate(sorted(kept_keys, key=lambda k: (-concept_images[k], k)))
    )

    predicate_images: Counter[str] = Counter()
    for graph in train_graphs:
        kept_objects = {
            obj.object_id
            for obj in graph.objects
            if not _is_junk(obj, junk_names, junk_keys) and concept_key(obj) in kept_keys
        }
        predicate_images.update({
            rel.predicate
            for rel in graph.relations
            if rel.subject_id in kept_objects and rel.object_id in kept_objects
        })
    kept_predicates = sorted(
        (p for p, n in predicate_images.items() if n >= min_train_images),
        key=lambda p: (-predicate_images[p], p),
    )
    predicates = tuple(
        Predicate(predicate_id=index, text=text, train_images=predicate_images[text])
        for index, text in enumerate(kept_predicates)
    )
    return GraphVocabulary(
        concepts=concepts, predicates=predicates, train_images=n_train, junk_concept_keys=junk_keys
    )


def _is_junk(obj: SceneObject, junk_names: frozenset[str], junk_keys: frozenset[str]) -> bool:
    return obj.name_raw in junk_names or concept_key(obj) in junk_keys


def _most_common_name(counts: Counter[str]) -> str:
    """Raw name used in the most images; ties go to the alphabetically first, so it is stable."""
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


@dataclass
class SplitCounts:
    """What was kept and dropped while mapping images onto the vocabulary."""

    images: int = 0
    objects: int = 0
    objects_kept: int = 0
    relations: int = 0
    triplets_kept: int = 0
    self_loop_triplets: int = 0
    images_without_concepts: int = 0
    images_without_triplets: int = 0
    dropped: Counter[GraphDropReason] = field(default_factory=Counter)

    def add(self, other: SplitCounts) -> None:
        """Accumulate another tally into this one."""
        self.images += other.images
        self.objects += other.objects
        self.objects_kept += other.objects_kept
        self.relations += other.relations
        self.triplets_kept += other.triplets_kept
        self.self_loop_triplets += other.self_loop_triplets
        self.images_without_concepts += other.images_without_concepts
        self.images_without_triplets += other.images_without_triplets
        self.dropped.update(other.dropped)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form with keep shares."""
        return {
            "images": self.images,
            "objects": self.objects,
            "objects_kept": self.objects_kept,
            "objects_kept_pct": _pct(self.objects_kept, self.objects),
            "relations": self.relations,
            "triplets_kept": self.triplets_kept,
            "relations_kept_pct": _pct(self.triplets_kept, self.relations),
            "self_loop_triplets": self.self_loop_triplets,
            "images_without_concepts": self.images_without_concepts,
            "images_without_triplets": self.images_without_triplets,
            "dropped": {str(reason): self.dropped.get(reason, 0) for reason in GraphDropReason},
        }


def build_image_graph(
    graph: SceneGraph, split: Split, vocabulary: GraphVocabulary, junk_names: frozenset[str]
) -> tuple[ImageGraph, SplitCounts]:
    """Map one scene graph onto the vocabulary: one node per concept, unique id triplets.

    Returns the graph and the tally of what this image kept and dropped.
    """
    return _map_image(
        graph, split, vocabulary.concept_ids(), vocabulary.predicate_ids(),
        junk_names, vocabulary.junk_concept_keys,
    )


def _map_image(
    graph: SceneGraph,
    split: Split,
    concept_ids: Mapping[str, int],
    predicate_ids: Mapping[str, int],
    junk_names: frozenset[str],
    junk_keys: frozenset[str],
) -> tuple[ImageGraph, SplitCounts]:
    counts = SplitCounts(images=1, objects=len(graph.objects), relations=len(graph.relations))
    concept_of_object: dict[int, int] = {}
    per_concept: Counter[int] = Counter()
    for obj in graph.objects:
        key = concept_key(obj)
        if obj.name_raw in junk_names:
            counts.dropped[GraphDropReason.OBJECT_JUNK_NAME] += 1
        elif key in junk_keys:
            counts.dropped[GraphDropReason.OBJECT_JUNK_CONCEPT] += 1
        elif key not in concept_ids:
            counts.dropped[GraphDropReason.OBJECT_OUT_OF_VOCABULARY] += 1
        else:
            concept_of_object[obj.object_id] = concept_ids[key]
            per_concept[concept_ids[key]] += 1
    counts.objects_kept = len(concept_of_object)

    triplets: set[tuple[int, int, int]] = set()
    for rel in graph.relations:
        subject = concept_of_object.get(rel.subject_id)
        target = concept_of_object.get(rel.object_id)
        predicate = predicate_ids.get(rel.predicate)
        if subject is None or target is None:
            counts.dropped[GraphDropReason.RELATION_ENDPOINT_DROPPED] += 1
        elif predicate is None:
            counts.dropped[GraphDropReason.RELATION_PREDICATE_OUT_OF_VOCABULARY] += 1
        elif (subject, predicate, target) in triplets:
            counts.dropped[GraphDropReason.RELATION_DUPLICATE] += 1
        else:
            triplets.add((subject, predicate, target))

    counts.triplets_kept = len(triplets)
    # Two objects of the same concept ("man next to man") share one node, hence a self loop.
    counts.self_loop_triplets = sum(1 for s, _, o in triplets if s == o)
    counts.images_without_concepts = int(not per_concept)
    counts.images_without_triplets = int(not triplets)
    ordered = sorted(per_concept)
    image_graph = ImageGraph(
        vg_image_id=graph.vg_image_id,
        split=split,
        concept_ids=tuple(ordered),
        concept_counts=tuple(per_concept[c] for c in ordered),
        triplets=tuple(sorted(triplets)),
    )
    return image_graph, counts


@dataclass(frozen=True)
class GraphBuild:
    """Result of the graph stage."""

    vocabulary: GraphVocabulary
    image_graphs: tuple[ImageGraph, ...]
    counts: dict[Split, SplitCounts]


def build_graphs(
    graphs: Iterable[SceneGraph],
    split_of: Mapping[int, Split],
    *,
    min_train_images: int,
    junk_names: frozenset[str],
) -> GraphBuild:
    """Build the vocabulary from train graphs, then map every graph onto it (sorted by image id).

    `graphs` and `split_of` must cover exactly the same images, each once.
    """
    ordered = sorted(graphs, key=lambda g: g.vg_image_id)
    ids = [g.vg_image_id for g in ordered]
    repeated = sorted(image_id for image_id, n in Counter(ids).items() if n > 1)
    if repeated:
        raise DataError(f"{len(repeated)} images have more than one scene graph, e.g. {repeated[:5]}")
    without_split = sorted(set(ids) - set(split_of))
    if without_split:
        raise DataError(f"{len(without_split)} scene graphs have no split, e.g. {without_split[:5]}")
    without_graph = sorted(set(split_of) - set(ids))
    if without_graph:
        raise DataError(f"{len(without_graph)} split images have no scene graph, e.g. {without_graph[:5]}")

    train = [g for g in ordered if split_of[g.vg_image_id] is Split.TRAIN]
    vocabulary = build_vocabulary(train, min_train_images=min_train_images, junk_names=junk_names)
    concept_ids, predicate_ids = vocabulary.concept_ids(), vocabulary.predicate_ids()
    counts = {split: SplitCounts() for split in Split}
    image_graphs = []
    for graph in ordered:
        split = split_of[graph.vg_image_id]
        image_graph, image_counts = _map_image(
            graph, split, concept_ids, predicate_ids, junk_names, vocabulary.junk_concept_keys
        )
        image_graphs.append(image_graph)
        counts[split].add(image_counts)
    return GraphBuild(vocabulary=vocabulary, image_graphs=tuple(image_graphs), counts=counts)


def graph_statistics(build: GraphBuild) -> dict[str, Any]:
    """Vocabulary size, hubs, per-split keep/drop counts and size distributions."""
    vocabulary = build.vocabulary
    texts = Counter(c.text for c in vocabulary.concepts)
    per_split = {}
    for split, counts in build.counts.items():
        graphs = [g for g in build.image_graphs if g.split is split]
        per_split[split.value] = {
            **counts.to_dict(),
            "concept_nodes": sum(len(g.concept_ids) for g in graphs),
            "concepts_per_image": _describe([len(g.concept_ids) for g in graphs]),
            "triplets_per_image": _describe([len(g.triplets) for g in graphs]),
        }
    return {
        "train_images": vocabulary.train_images,
        "concepts": len(vocabulary.concepts),
        "concepts_with_synset": sum(1 for c in vocabulary.concepts if c.synset is not None),
        "predicates": len(vocabulary.predicates),
        "junk_concepts": sorted(vocabulary.junk_concept_keys),
        # Degree of a concept node in the image–concept graph, over train images.
        "train_images_per_concept": _describe([c.train_images for c in vocabulary.concepts]),
        # Distinct concepts can share a display text (two synsets both called "man" by annotators).
        "concept_texts_shared": sum(n for n in texts.values() if n > 1),
        "shared_text_examples": sorted(t for t, n in texts.items() if n > 1)[:_TOP_HUBS],
        "top_concepts_by_train_image_pct": [
            (c.text, _pct(c.train_images, vocabulary.train_images))
            for c in vocabulary.concepts[:_TOP_HUBS]
        ],
        "top_predicates_by_train_image_pct": [
            (p.text, _pct(p.train_images, vocabulary.train_images))
            for p in vocabulary.predicates[:_TOP_HUBS]
        ],
        "by_split": per_split,
    }


def _describe(values: list[int]) -> dict[str, float]:
    """`describe`, or an empty mapping when the sample is too small for quantiles."""
    return describe(values) if len(values) >= 2 else {}


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0
