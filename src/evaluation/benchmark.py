"""Evaluation data for one split and scoring of a method on it. No file I/O here."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.metrics import DEFAULT_KS, ranks_of_relevant, retrieval_metrics


@dataclass(frozen=True)
class SplitData:
    """Gallery images and queries of one split, with stored embeddings in matching order."""

    split: Split
    image_ids: tuple[int, ...]
    query_ids: tuple[str, ...]
    relevant_columns: np.ndarray  # for each query, the gallery column of its relevant image
    image_vectors: np.ndarray
    query_vectors: np.ndarray


def build_split_data(
    split: Split,
    images: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    image_table: EmbeddingTable,
    query_table: EmbeddingTable,
) -> SplitData:
    """Select the split's images (the gallery) and queries, sorted by id.

    The gallery holds only images of this split, so a method is never scored against images
    it was trained on. Every query must have exactly one relevant image, inside the gallery.
    """
    image_ids = tuple(sorted(int(r["vg_image_id"]) for r in images if r["split"] == split.value))
    selected = sorted((q for q in queries if q["split"] == split.value), key=lambda q: q["query_id"])
    if not image_ids or not selected:
        raise DataError(f"split '{split.value}' has {len(image_ids)} images and {len(selected)} queries")
    column_of = {image_id: column for column, image_id in enumerate(image_ids)}
    columns = []
    for query in selected:
        relevant = query["relevant_ids"]
        if len(relevant) != 1 or int(relevant[0]) not in column_of:
            raise DataError(
                f"query {query['query_id']} must have one relevant image in split '{split.value}', got {relevant}"
            )
        columns.append(column_of[int(relevant[0])])
    query_ids = tuple(q["query_id"] for q in selected)
    return SplitData(
        split=split,
        image_ids=image_ids,
        query_ids=query_ids,
        relevant_columns=np.asarray(columns),
        image_vectors=image_table.select(image_ids),
        query_vectors=query_table.select(query_ids),
    )


def clip_scores(data: SplitData) -> np.ndarray:
    """Cosine of every query with every gallery image (rows are unit length)."""
    return data.query_vectors @ data.image_vectors.T


def score_method(scores: np.ndarray, data: SplitData, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Recall@K and MRR of a (queries × gallery) score matrix on this split."""
    expected = (len(data.query_ids), len(data.image_ids))
    if scores.shape != expected:
        raise DataError(f"score matrix has shape {scores.shape}, expected {expected}")
    return retrieval_metrics(ranks_of_relevant(scores, data.relevant_columns), ks)


def graph_vector_scores(data: SplitData, graph_table: EmbeddingTable) -> np.ndarray:
    """Cosine of every query with the graph vector of every gallery image (rows are unit length)."""
    return data.query_vectors @ graph_table.select(data.image_ids).T


def aggregate_runs(runs: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each metric over runs (std is 0 for a single run)."""
    if not runs:
        raise DataError("no runs to aggregate")
    names = set(runs[0])
    if any(set(run) != names for run in runs):
        raise DataError("runs report different metrics")
    return {
        name: {
            "mean": float(np.mean([run[name] for run in runs])),
            "std": float(np.std([run[name] for run in runs], ddof=1)) if len(runs) > 1 else 0.0,
        }
        for name in sorted(names)
    }
