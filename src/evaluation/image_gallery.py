"""Text-to-image retrieval over the images of the entities: the gallery is every image.

An image is correct for a query when it shows one of the query's answer entities. Each
(query, answer entity) pair gets one rank: that of the entity's best image among the images
of entities that are not answers. Other correct images are left out, so a query is not
punished for ranking a different right image first.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.mkg_w_benchmark import EntitySplit


def best_image_ranks(
    scores: np.ndarray, answer_images: Sequence[Sequence[Sequence[int]]], known: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """1-based rank of the best image of every (query, answer entity) pair.

    `answer_images[i]` holds, per answer entity of query i, the columns of its images;
    `known[i]` are columns of other correct images. Ties count against the answer. Returns
    (ranks, query row of each rank), one entry per pair, in query order.
    """
    if scores.ndim != 2 or len(answer_images) != scores.shape[0] or len(known) != scores.shape[0]:
        raise DataError(f"scores {scores.shape} do not match {len(answer_images)} answer / {len(known)} known lists")
    if not np.isfinite(scores).all():
        raise DataError("scores contain NaN or infinite values")
    ranks: list[int] = []
    rows: list[int] = []
    for row, (groups, others) in enumerate(zip(answer_images, known, strict=True)):
        if not groups or not all(groups):
            raise DataError(f"query row {row} has an answer without images")
        correct = np.fromiter({*(column for group in groups for column in group), *others}, dtype=np.int64)
        if correct.min() < 0 or correct.max() >= scores.shape[1]:
            raise DataError(f"query row {row} names a column outside the score matrix")
        line = scores[row]
        for group in groups:
            best = line[list(group)].max()
            ranks.append(int((line >= best).sum()) - int((line[correct] >= best).sum()) + 1)
            rows.append(row)
    return np.asarray(ranks), np.asarray(rows)


@dataclass(frozen=True)
class ImageSplit(EntitySplit):
    """Queries of one split scored against every image; `relevant` and `known` are image columns."""

    answer_images: tuple[tuple[tuple[int, ...], ...], ...]  # per query, per answer entity: its image columns
    image_entities: tuple[int, ...]  # entity id shown by each gallery column

    def ranks(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rank of the best image of every (query, answer entity) pair under `scores`."""
        return best_image_ranks(scores, self.answer_images, self.known)


def build_image_split(
    split: Split,
    entities: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    image_table: EmbeddingTable,
    query_table: EmbeddingTable,
) -> ImageSplit:
    """Select the split's queries (sorted by id) and score them against every image by cosine."""
    files = [(file, e["entity_id"]) for e in entities for file in e["image_files"]]
    columns_of: dict[int, list[int]] = {}
    for column, (_, entity_id) in enumerate(files):
        columns_of.setdefault(entity_id, []).append(column)
    selected = tuple(sorted((q for q in queries if q["split"] == split.value), key=lambda q: q["query_id"]))
    answer_images = tuple(tuple(tuple(columns_of[i]) for i in q["relevant_ids"]) for q in selected)
    return ImageSplit(
        split=split,
        queries=selected,
        clip_scores=query_table.select([q["query_id"] for q in selected])
        @ image_table.select([file for file, _ in files]).T,
        relevant=tuple(tuple(column for group in groups for column in group) for groups in answer_images),
        # Known answers without images are not in the gallery, so there is nothing to skip.
        known=tuple(tuple(column for i in q["known_ids"] for column in columns_of.get(i, ())) for q in selected),
        answer_images=answer_images,
        image_entities=tuple(entity_id for _, entity_id in files),
    )
