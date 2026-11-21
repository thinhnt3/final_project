"""Pieces shared by the MKG-W training scripts: graph variants, train queries, seed summaries."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import torch

from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.benchmark import aggregate_runs
from src.models.kg_train import KgQuerySet
from src.retrieval.path_score import shuffled_tails

# Report segments printed after a run; every segment is in the JSON.
LOGGED_PREFIXES = ("all", "group=", "hops=")
_METRICS = ("recall@1", "recall@10", "mrr")


class Variant(StrEnum):
    """Which graph the encoder sees."""

    REAL = "real"
    SHUFFLED_EDGES = "shuffled_edges"
    NO_EDGES = "no_edges"


def variant_triples(variant: Variant, train: Sequence[Sequence[int]], seed: int) -> list[tuple[int, int, int]]:
    """The (head, relation, tail) graph a variant trains on, built from the train triples."""
    triples = [(head, relation, tail) for head, relation, tail in train]
    if variant is Variant.NO_EDGES:
        return []
    if variant is Variant.SHUFFLED_EDGES:
        pairs = shuffled_tails([(head, tail) for head, _, tail in triples], seed)
        return [(head, relation, tail) for (head, tail), (_, relation, _) in zip(pairs, triples, strict=True)]
    return triples


def train_query_set(
    queries: Sequence[Mapping[str, Any]],
    query_table: EmbeddingTable,
    column_of: Mapping[int, int],
    triple_row: Mapping[tuple[int, int, int], int],
) -> KgQuerySet:
    """Train queries with the rows of the graph triples that state each query's own fact."""
    selected = sorted((q for q in queries if q["split"] == Split.TRAIN.value), key=lambda q: q["query_id"])
    own_triples = []
    for query in selected:
        rows = []
        for answer in query["relevant_ids"] if query["anchor_id"] is not None else ():
            head, tail = (query["anchor_id"], answer) if query["direction"] == "tail" else (answer, query["anchor_id"])
            # Absent in the shuffled and empty graphs: there the fact is not an edge, so nothing to remove.
            if (head, query["relation_id"], tail) in triple_row:
                rows.append(triple_row[(head, query["relation_id"], tail)])
        own_triples.append(rows)
    return KgQuerySet(
        query_vectors=torch.from_numpy(query_table.select([q["query_id"] for q in selected])),
        relevant=[[column_of[i] for i in q["relevant_ids"]] for q in selected],
        known=[[column_of[i] for i in q["known_ids"] if i in column_of] for q in selected],
        own_triples=own_triples,
    )


def seed_summary(per_seed: Sequence[Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """Mean and std over seeds of each metric, per report segment."""
    return {
        name: {"answers": per_seed[0][name]["answers"],
               **aggregate_runs([{m: seed[name][m] for m in _METRICS} for seed in per_seed])}
        for name in per_seed[0]
    }
