"""Entity-level retrieval scoring: several images per entity, several correct entities per query.

An entity's score for a query is the best score among its images. Ranks are filtered, as in
link prediction: when ranking one correct entity, the query's other correct entities are
left out, so a query is not punished for ranking a different right answer first.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.exceptions import DataError


def entity_scores(image_scores: np.ndarray, group_starts: np.ndarray) -> np.ndarray:
    """Collapse a (queries × images) matrix to (queries × entities) by max over each entity's images.

    Images of one entity must be adjacent columns; `group_starts[i]` is the first column of
    entity i, strictly increasing from 0.
    """
    starts = np.asarray(group_starts)
    if image_scores.ndim != 2 or starts.ndim != 1 or starts.size == 0:
        raise DataError(f"need a 2-D score matrix and 1-D group starts, got {image_scores.shape}, {starts.shape}")
    if starts[0] != 0 or (np.diff(starts) <= 0).any() or starts[-1] >= image_scores.shape[1]:
        raise DataError("group starts must begin at 0, increase strictly and stay inside the matrix")
    return np.maximum.reduceat(image_scores, starts, axis=1)


def best_image_scores(
    query_vectors: np.ndarray, image_vectors: np.ndarray, group_starts: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """(queries × entities) cosine of each query with the entity's best image; rows must be unit length.

    Computed in query chunks so the (queries × images) matrix is never held in full.
    """
    parts = [
        entity_scores(query_vectors[start:start + chunk] @ image_vectors.T, group_starts)
        for start in range(0, query_vectors.shape[0], chunk)
    ]
    return np.concatenate(parts)


def filtered_ranks(
    scores: np.ndarray, relevant: Sequence[Sequence[int]], known: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """1-based filtered rank of every (query, relevant column) pair.

    `relevant[i]` are the columns scored for query i; `known[i]` are other correct columns
    that are skipped. Ties count against the relevant column. Returns (ranks, query row of
    each rank), one entry per pair, in query order.
    """
    if scores.ndim != 2 or len(relevant) != scores.shape[0] or len(known) != scores.shape[0]:
        raise DataError(f"scores {scores.shape} do not match {len(relevant)} relevant / {len(known)} known lists")
    if not np.isfinite(scores).all():
        raise DataError("scores contain NaN or infinite values")
    ranks: list[int] = []
    rows: list[int] = []
    for row, (answers, others) in enumerate(zip(relevant, known, strict=True)):
        if not answers:
            raise DataError(f"query row {row} has no relevant column")
        correct = np.fromiter({*answers, *others}, dtype=np.int64)
        if correct.min() < 0 or correct.max() >= scores.shape[1]:
            raise DataError(f"query row {row} names a column outside the score matrix")
        line = scores[row]
        for answer in answers:
            at_least = int((line >= line[answer]).sum())
            # Remove the other correct columns that also score at least as high.
            skipped = int((line[correct] >= line[answer]).sum()) - 1
            ranks.append(at_least - skipped)
            rows.append(row)
    return np.asarray(ranks), np.asarray(rows)
