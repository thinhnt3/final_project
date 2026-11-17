import numpy as np
import pytest

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable, l2_normalize
from src.evaluation.benchmark import (
    aggregate_runs,
    build_split_data,
    clip_scores,
    graph_vector_scores,
    score_method,
)


def _tables():
    # Image i and its caption share axis i, so each caption matches its own image.
    images = EmbeddingTable(ids=(30, 10, 20, 40), vectors=np.eye(4, dtype=np.float32)[[2, 0, 1, 3]])
    queries = EmbeddingTable(ids=("q-20", "q-10", "q-30", "q-40"), vectors=np.eye(4, dtype=np.float32)[[1, 0, 2, 3]])
    return images, queries


IMAGES = [
    {"vg_image_id": 10, "split": "test"},
    {"vg_image_id": 20, "split": "test"},
    {"vg_image_id": 30, "split": "train"},
    {"vg_image_id": 40, "split": "val"},
]
QUERIES = [
    {"query_id": "q-20", "relevant_ids": [20], "split": "test"},
    {"query_id": "q-10", "relevant_ids": [10], "split": "test"},
    {"query_id": "q-30", "relevant_ids": [30], "split": "train"},
    {"query_id": "q-40", "relevant_ids": [40], "split": "val"},
]


def test_split_data_holds_only_that_split_in_sorted_order():
    data = build_split_data(Split.TEST, IMAGES, QUERIES, *_tables())
    assert data.image_ids == (10, 20)  # train image 30 and val image 40 are not in the gallery
    assert data.query_ids == ("q-10", "q-20")
    assert data.relevant_columns.tolist() == [0, 1]
    assert data.image_vectors.shape == (2, 4) and data.query_vectors.shape == (2, 4)


def test_clip_scores_rank_each_caption_with_its_own_image_first():
    data = build_split_data(Split.TEST, IMAGES, QUERIES, *_tables())
    scores = clip_scores(data)
    assert scores.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert score_method(scores, data) == {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0}


def test_embedding_rows_follow_ids_not_table_order():
    images, queries = _tables()
    shuffled = EmbeddingTable(ids=images.ids[::-1], vectors=images.vectors[::-1].copy())
    data = build_split_data(Split.TEST, IMAGES, QUERIES, shuffled, queries)
    assert score_method(clip_scores(data), data)["recall@1"] == 1.0


def test_query_whose_image_is_in_another_split_is_rejected():
    queries = [*QUERIES, {"query_id": "q-x", "relevant_ids": [30], "split": "test"}]
    images, table = _tables()
    extended = EmbeddingTable(ids=(*table.ids, "q-x"), vectors=l2_normalize(np.vstack([table.vectors, np.ones(4)])))
    with pytest.raises(DataError, match="one relevant image in split 'test'"):
        build_split_data(Split.TEST, IMAGES, queries, images, extended)


def test_empty_split_and_wrong_score_shape_are_rejected():
    images, queries = _tables()
    with pytest.raises(DataError, match="has 0 images"):
        build_split_data(Split.TEST, [r for r in IMAGES if r["split"] != "test"], QUERIES, images, queries)
    data = build_split_data(Split.TEST, IMAGES, QUERIES, images, queries)
    with pytest.raises(DataError, match="expected"):
        score_method(np.zeros((2, 3)), data)


def test_graph_vector_scores_follow_the_gallery_order_of_the_split():
    images, queries = _tables()
    data = build_split_data(Split.TEST, IMAGES, QUERIES, images, queries)  # gallery (10, 20), queries q-10, q-20
    # Graph vectors stored in another order than the gallery; image 20 lies on axis 0, image 10 on axis 1.
    graph_table = EmbeddingTable(ids=(20, 30, 10), vectors=np.eye(4, dtype=np.float32)[[1, 3, 0]])
    scores = graph_vector_scores(data, graph_table)
    assert scores.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(DataError, match="no embedding"):
        graph_vector_scores(data, EmbeddingTable(ids=(20,), vectors=np.eye(4, dtype=np.float32)[:1]))


def test_aggregate_runs_reports_mean_and_sample_std():
    summary = aggregate_runs(
        [{"recall@1": 0.40, "mrr": 0.5}, {"recall@1": 0.44, "mrr": 0.5}, {"recall@1": 0.42, "mrr": 0.5}]
    )
    assert summary["recall@1"]["mean"] == pytest.approx(0.42)
    assert summary["recall@1"]["std"] == pytest.approx(0.02)
    assert summary["mrr"] == {"mean": 0.5, "std": 0.0}
    assert aggregate_runs([{"mrr": 0.3}]) == {"mrr": {"mean": 0.3, "std": 0.0}}


def test_aggregate_runs_rejects_empty_and_mismatched_input():
    with pytest.raises(DataError, match="no runs"):
        aggregate_runs([])
    with pytest.raises(DataError, match="different metrics"):
        aggregate_runs([{"mrr": 0.3}, {"recall@1": 0.3}])
