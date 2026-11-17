"""Retrieval metrics for queries with exactly one relevant item: Recall@K and MRR."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.exceptions import DataError

DEFAULT_KS = (1, 5, 10)


def ranks_of_relevant(scores: np.ndarray, relevant_columns: Sequence[int] | np.ndarray) -> np.ndarray:
    """1-based rank of each query's relevant item in a (queries × items) score matrix.

    Ties are resolved against the relevant item: every other item with an equal score is
    ranked above it, so a method cannot gain from the order of columns.
    """
    columns = np.asarray(relevant_columns)
    if scores.ndim != 2 or columns.shape != (scores.shape[0],):
        raise DataError(f"scores {scores.shape} do not match relevant columns of shape {columns.shape}")
    if scores.shape[0] == 0:
        raise DataError("no queries to rank")
    if columns.min() < 0 or columns.max() >= scores.shape[1]:
        raise DataError("a relevant column is outside the score matrix")
    if not np.isfinite(scores).all():
        raise DataError("scores contain NaN or infinite values")
    relevant_scores = scores[np.arange(scores.shape[0]), columns]
    # Counts the relevant item itself once (its score is >= itself), which makes the rank 1-based.
    return (scores >= relevant_scores[:, None]).sum(axis=1)


def retrieval_metrics(ranks: np.ndarray, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Recall@K for each K and mean reciprocal rank, from 1-based ranks."""
    if ranks.size == 0:
        raise DataError("no ranks to score")
    if ranks.min() < 1:
        raise DataError("ranks must be 1-based")
    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / ranks).mean())
    return metrics
