"""Filtered link prediction ranks both ends of a triple and skips other true completions."""
from __future__ import annotations

import numpy as np

from src.evaluation.link_prediction import known_completions, link_prediction_metrics


def test_other_true_tails_are_skipped() -> None:
    # Entity 0 has two true tails (1 and 2); a scorer that puts 1 above 2 must still give 2 rank 1.
    triples = [[0, 0, 1], [0, 0, 2]]
    known = known_completions(triples)
    assert known[(0, 0, True)] == {1, 2}
    assert known[(2, 0, False)] == {0}

    def score(anchors: np.ndarray, relations: np.ndarray, anchor_is_head: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(anchors), 4), dtype=np.float32)
        scores[anchor_is_head, 1], scores[anchor_is_head, 2] = 3.0, 2.0  # tails of entity 0
        scores[~anchor_is_head, 0] = 1.0  # the head of both triples
        return scores

    metrics = link_prediction_metrics(score, triples, known, batch_size=3)
    assert metrics == {"mrr": 1.0, "hits@1": 1.0, "hits@3": 1.0, "hits@10": 1.0}


def test_a_wrong_scorer_gets_low_ranks() -> None:
    triples = [[0, 0, 1]]

    def score(anchors: np.ndarray, relations: np.ndarray, anchor_is_head: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.0, 0.0, 5.0, 4.0], dtype=np.float32), (len(anchors), 1))

    metrics = link_prediction_metrics(score, triples, known_completions(triples), batch_size=8)
    # Tail prediction: entity 1 ties entity 0 below two others, rank 4. Head prediction: entity 0, rank 4.
    assert metrics["hits@3"] == 0.0
    assert metrics["mrr"] == 0.25
