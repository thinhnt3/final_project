import numpy as np
import pytest

from src.core.exceptions import DataError
from src.retrieval.path_score import normalized_adjacency, path_scores, shuffled_tails

# 0 — 1 — 2 — 3 chain, plus isolated node 4.
CHAIN = [(0, 1), (1, 2), (2, 3)]


def test_normalized_adjacency_is_symmetric_and_degree_scaled():
    matrix = normalized_adjacency([*CHAIN, (1, 0)], 5).toarray()  # the reversed duplicate counts once
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 1] == pytest.approx(1 / np.sqrt(1 * 2))
    assert matrix[1, 2] == pytest.approx(1 / np.sqrt(2 * 2))
    assert matrix[4].sum() == 0


def test_path_scores_reach_exactly_max_hops():
    adjacency = normalized_adjacency(CHAIN, 5)
    one_hop = path_scores(adjacency, [0], max_hops=1, decay=0.5)[0]
    assert one_hop.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    three_hops = path_scores(adjacency, [0], max_hops=3, decay=0.5)[0]
    assert three_hops[0] == 0.0  # the anchor itself is never an answer
    assert three_hops[1] == 1.0 and three_hops[1] > three_hops[2] > three_hops[3] > 0
    assert three_hops[4] == 0.0


def test_anchor_without_edges_scores_nothing():
    assert not path_scores(normalized_adjacency(CHAIN, 5), [4], max_hops=3, decay=0.5).any()


def test_bad_inputs_fail():
    with pytest.raises(DataError, match="outside"):
        normalized_adjacency([(0, 9)], 5)
    with pytest.raises(DataError, match="decay"):
        path_scores(normalized_adjacency(CHAIN, 5), [0], max_hops=3, decay=0.0)


def test_shuffled_tails_keep_heads_and_tail_counts():
    edges = [(i, i + 100) for i in range(50)]
    shuffled = shuffled_tails(edges, seed=13)
    assert [h for h, _ in shuffled] == [h for h, _ in edges]
    assert sorted(t for _, t in shuffled) == sorted(t for _, t in edges)
    assert shuffled != edges and shuffled == shuffled_tails(edges, seed=13)
