"""Ranks and splits of the image gallery."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.image_gallery import best_image_ranks, build_image_split


def test_rank_uses_best_image_and_skips_other_correct_images() -> None:
    # Columns 0-1 show answer A, 2 shows answer B, 3 is a known answer, 4-5 are wrong.
    scores = np.array([[0.2, 0.6, 0.9, 0.8, 0.7, 0.1]])
    ranks, rows = best_image_ranks(scores, [[[0, 1], [2]]], [[3]])
    # A's best image (0.6) is behind one wrong image (0.7); B, the known image and A's other image are skipped.
    assert ranks.tolist() == [2, 1]
    assert rows.tolist() == [0, 0]


def test_tie_with_wrong_image_counts_against_answer() -> None:
    ranks, _ = best_image_ranks(np.array([[0.5, 0.5]]), [[[0]]], [[]])
    assert ranks.tolist() == [2]


def test_answer_without_images_is_rejected() -> None:
    with pytest.raises(DataError):
        best_image_ranks(np.array([[0.5, 0.4]]), [[[]]], [[]])


def test_build_image_split_maps_entities_to_image_columns() -> None:
    entities = [
        {"entity_id": 0, "image_files": ["a.jpg", "b.jpg"]},
        {"entity_id": 1, "image_files": []},
        {"entity_id": 2, "image_files": ["c.jpg"]},
    ]
    queries = [
        {"query_id": "q1", "split": "test", "relevant_ids": [2], "known_ids": [0, 1]},
        {"query_id": "q0", "split": "val", "relevant_ids": [0], "known_ids": []},
    ]
    images = EmbeddingTable(ids=("a.jpg", "b.jpg", "c.jpg"), vectors=np.eye(3, dtype=np.float32))
    query_table = EmbeddingTable(ids=("q0", "q1"), vectors=np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))

    split = build_image_split(Split.TEST, entities, queries, images, query_table)

    assert [q["query_id"] for q in split.queries] == ["q1"]
    assert split.image_entities == (0, 0, 2)
    assert split.answer_images == (((2,),),)
    assert split.known == ((0, 1),)
    assert split.clip_scores.tolist() == [[0.0, 0.0, 1.0]]
    assert split.ranks(split.clip_scores)[0].tolist() == [1]
