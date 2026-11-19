"""Assemble the graphs and caption vectors of one split for training and evaluation."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.data.text_match import names, names_any
from src.embeddings.store import EmbeddingTable
from src.models.graph_data import image_graph_to_data
from src.models.train import RetrievalSet


def build_retrieval_set(
    split: Split,
    images: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    image_graphs: Sequence[Mapping[str, Any]],
    image_table: EmbeddingTable,
    query_table: EmbeddingTable,
) -> tuple[RetrievalSet, tuple[int, ...]]:
    """Sub-graphs and caption vectors of a split's images, sorted by image id; also the ids.

    Captions of an image are ordered by query id. Every image must have a graph and the same
    number of captions.
    """
    image_ids = tuple(sorted(int(r["vg_image_id"]) for r in images if r["split"] == split.value))
    if not image_ids:
        raise DataError(f"split '{split.value}' has no images")
    graph_of = {int(g["vg_image_id"]): g for g in image_graphs}
    without_graph = [image_id for image_id in image_ids if image_id not in graph_of]
    if without_graph:
        raise DataError(f"{len(without_graph)} images of split '{split.value}' have no graph, e.g. {without_graph[:5]}")

    queries_of: dict[int, list[str]] = defaultdict(list)
    for query in queries:
        if query["split"] == split.value:
            queries_of[int(query["relevant_ids"][0])].append(query["query_id"])
    counts = {len(queries_of[image_id]) for image_id in image_ids}
    if len(counts) != 1 or 0 in counts:
        raise DataError(
            f"images of split '{split.value}' must all have the same, non-zero number of captions, got {counts}"
        )

    image_row = image_table.row_of()
    graphs = [image_graph_to_data(graph_of[image_id], image_row[image_id]) for image_id in image_ids]
    query_ids = [query_id for image_id in image_ids for query_id in sorted(queries_of[image_id])]
    vectors = torch.from_numpy(query_table.select(query_ids)).reshape(len(image_ids), counts.pop(), -1)
    return RetrievalSet(graphs=graphs, caption_vectors=vectors), image_ids


def hard_negative_partners(
    image_graphs: Sequence[Mapping[str, Any]], image_ids: Sequence[int], limit: int
) -> tuple[tuple[int, ...], ...]:
    """For each image (by position in `image_ids`), up to `limit` positions of hard negatives.

    A hard negative of an image holds both objects of one of its triplets but not that
    triplet, so only the relation tells the two apart. Rarer object pairs are used first,
    so hub pairs such as (man, shirt) do not fill every list.
    """
    graph_of = {int(g["vg_image_id"]): g for g in image_graphs}
    concepts = [{int(c) for c in graph_of[i]["concept_ids"]} for i in image_ids]
    triplets = [{(int(s), int(p), int(o)) for s, p, o in graph_of[i]["triplets"]} for i in image_ids]
    holders: dict[int, set[int]] = defaultdict(set)
    for position, ids in enumerate(concepts):
        for concept in ids:
            holders[concept].add(position)
    partners = []
    for position, own in enumerate(triplets):
        by_pair = sorted(
            ((holders[s] & holders[o], (s, p, o)) for s, p, o in own if s != o), key=lambda x: (len(x[0]), x[1])
        )
        chosen: list[int] = []
        for others, triplet in by_pair:
            for other in sorted(others - {position} - set(chosen)):
                if triplet not in triplets[other]:
                    chosen.append(other)
                    if len(chosen) == limit:
                        break
            if len(chosen) == limit:
                break
        partners.append(tuple(chosen))
    return tuple(partners)


def relation_mentions(
    image_graphs: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    queries: Sequence[Mapping[str, Any]],
    concept_names: Mapping[int, Sequence[str]],
    predicate_text: Mapping[int, str],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Per image (in `image_ids` order), per caption (by query id), the relation nodes it names.

    A caption names a triplet when it contains the predicate and one name of each object.
    Node indices follow `image_graph_to_data`: image, objects, then one node per triplet.
    """
    graph_of = {int(g["vg_image_id"]): g for g in image_graphs}
    captions: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for query in queries:
        captions[int(query["relevant_ids"][0])].append((query["query_id"], query["text"]))
    result = []
    for image_id in image_ids:
        graph = graph_of[image_id]
        first_relation = 1 + len(graph["concept_ids"])
        per_caption = []
        for _, text in sorted(captions[image_id]):
            named = tuple(
                first_relation + position
                for position, (s, p, o) in enumerate(graph["triplets"])
                if int(s) != int(o)
                and names(text, predicate_text[int(p)])
                and names_any(text, concept_names[int(s)])
                and names_any(text, concept_names[int(o)])
            )
            per_caption.append(named)
        result.append(tuple(per_caption))
    return tuple(result)


def subset_retrieval_set(
    data: RetrievalSet, image_ids: Sequence[int], fraction: float, seed: int
) -> tuple[RetrievalSet, tuple[int, ...]]:
    """A fixed random share of the images, in their original order; the draw ignores the training seed."""
    if not 0 < fraction <= 1:
        raise DataError(f"train fraction must lie in (0, 1], got {fraction}")
    if fraction == 1:
        return data, tuple(image_ids)
    keep = max(2, round(len(image_ids) * fraction))
    chosen = sorted(torch.randperm(len(image_ids), generator=torch.Generator().manual_seed(seed))[:keep].tolist())
    return (
        RetrievalSet(graphs=[data.graphs[i] for i in chosen], caption_vectors=data.caption_vectors[chosen]),
        tuple(image_ids[i] for i in chosen),
    )
