"""Shared pieces of the MKG-W benchmarks: the entity gallery, split scores and report segments."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.entity_retrieval import best_image_scores, filtered_ranks
from src.evaluation.metrics import retrieval_metrics
from src.retrieval.rerank import rerank_scores

# Metrics are compared after rounding so that float noise cannot decide the selection.
_SELECTION_DECIMALS = 6


@dataclass(frozen=True)
class EntitySplit:
    """Queries of one split with their CLIP scores against the entity gallery."""

    split: Split
    queries: tuple[Mapping[str, Any], ...]
    clip_scores: np.ndarray  # (queries × gallery entities)
    relevant: tuple[tuple[int, ...], ...]  # gallery columns of each query's answers
    known: tuple[tuple[int, ...], ...]  # gallery columns to skip when ranking

    def ranks(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filtered ranks of every (query, answer) pair under `scores`, and each pair's query row."""
        return filtered_ranks(scores, self.relevant, self.known)


def gallery_columns(entities: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Entity id → gallery column, for entities that have images, in entity order."""
    return {e["entity_id"]: column for column, e in enumerate(e for e in entities if e["image_files"])}


def build_entity_split(
    split: Split,
    entities: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    image_table: EmbeddingTable,
    query_table: EmbeddingTable,
) -> EntitySplit:
    """Select the split's queries (sorted by id) and score them against every entity with images."""
    gallery = [e for e in entities if e["image_files"]]
    column_of = gallery_columns(entities)
    selected = tuple(sorted((q for q in queries if q["split"] == split.value), key=lambda q: q["query_id"]))
    group_starts = np.cumsum([0] + [len(e["image_files"]) for e in gallery[:-1]])
    return EntitySplit(
        split=split,
        queries=selected,
        clip_scores=best_image_scores(
            query_table.select([q["query_id"] for q in selected]),
            image_table.select([file for e in gallery for file in e["image_files"]]),
            group_starts,
        ),
        relevant=tuple(tuple(column_of[i] for i in q["relevant_ids"]) for q in selected),
        # Known answers without images are not in the gallery, so there is nothing to skip.
        known=tuple(tuple(column_of[i] for i in q["known_ids"] if i in column_of) for q in selected),
    )


def select_alpha_top_k(
    val: EntitySplit, graph_scores: np.ndarray, alphas: Sequence[float], top_ks: Sequence[int]
) -> dict[str, float]:
    """Best (alpha, top_k) on validation for `alpha·CLIP + (1−alpha)·graph` over the top-k CLIP entities.

    Highest Recall@1, then MRR. On a tie the setting closest to plain CLIP wins (highest
    alpha, then smallest top_k), so a graph score that adds nothing is not reported as a gain.
    """
    best_key: tuple[float, float, float, int] | None = None
    best: dict[str, float] = {}
    for top_k in top_ks:
        for alpha in alphas:
            ranks, _ = val.ranks(rerank_scores(val.clip_scores, graph_scores, alpha, top_k))
            metrics = retrieval_metrics(ranks, (1,))
            key = (round(metrics["recall@1"], _SELECTION_DECIMALS), round(metrics["mrr"], _SELECTION_DECIMALS),
                   alpha, -top_k)
            if best_key is None or key > best_key:
                best_key = key
                best = {"alpha": alpha, "top_k": top_k,
                        "val_recall@1": round(metrics["recall@1"], 4), "val_mrr": round(metrics["mrr"], 4)}
    return best


def segment_names(query: Mapping[str, Any], relations: Sequence[Mapping[str, Any]], max_hops: int) -> list[str]:
    """Names of the report segments a query belongs to."""
    names = ["all", f"group={query['group']}", f"kind={query['kind']}"]
    if query["relation_id"] is not None:
        names.append(f"hops={query['hops'] or f'over_{max_hops}_or_none'}")
        names.append(f"relation={relations[query['relation_id']]['pid']}/{query['direction']}")
    return names


def segment_metrics(
    data: EntitySplit,
    scores: np.ndarray,
    relations: Sequence[Mapping[str, Any]],
    max_hops: int,
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Recall@K and MRR per report segment; `answers` is the number of (query, answer) pairs scored."""
    ranks, rows = data.ranks(scores)
    pairs_of: dict[str, list[int]] = defaultdict(list)
    for pair, row in enumerate(rows):
        for name in segment_names(data.queries[row], relations, max_hops):
            pairs_of[name].append(pair)
    return {
        name: {"answers": len(pairs), **{k: round(v, 4) for k, v in retrieval_metrics(ranks[pairs], ks).items()}}
        for name, pairs in sorted(pairs_of.items())
    }
