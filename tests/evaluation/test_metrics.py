import numpy as np
import pytest

from src.core.exceptions import DataError
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics


def test_rank_counts_items_scored_above_the_relevant_one():
    scores = np.array([
        [0.9, 0.1, 0.5],   # relevant col 0 → rank 1
        [0.9, 0.1, 0.5],   # relevant col 2 → rank 2
        [0.9, 0.1, 0.5],   # relevant col 1 → rank 3
    ])
    assert ranks_of_relevant(scores, [0, 2, 1]).tolist() == [1, 2, 3]


def test_ties_count_against_the_relevant_item():
    scores = np.array([[0.5, 0.5, 0.5, 0.1]])
    # Two other items share the relevant score, so the relevant item is ranked third,
    # whichever column it sits in.
    assert ranks_of_relevant(scores, [0]).tolist() == [3]
    assert ranks_of_relevant(scores, [2]).tolist() == [3]
    assert ranks_of_relevant(np.zeros((1, 4)), [1]).tolist() == [4]


def test_metrics_at_rank_boundaries():
    ranks = np.array([1, 5, 6, 10, 11])
    metrics = retrieval_metrics(ranks)
    assert metrics["recall@1"] == pytest.approx(1 / 5)
    assert metrics["recall@5"] == pytest.approx(2 / 5)   # rank 5 counts, rank 6 does not
    assert metrics["recall@10"] == pytest.approx(4 / 5)  # rank 10 counts, rank 11 does not
    assert metrics["mrr"] == pytest.approx((1 + 1 / 5 + 1 / 6 + 1 / 10 + 1 / 11) / 5)


def test_custom_ks():
    assert retrieval_metrics(np.array([2, 3]), ks=(2,)) == {"recall@2": 0.5, "mrr": pytest.approx((1 / 2 + 1 / 3) / 2)}


@pytest.mark.parametrize(
    ("scores", "columns", "message"),
    [
        (np.zeros((2, 3)), [0], "do not match"),
        (np.zeros((1, 3)), [3], "outside"),
        (np.zeros((1, 3)), [-1], "outside"),
        (np.array([[np.nan, 0.0]]), [0], "NaN"),
        (np.zeros((0, 3)), [], "no queries"),
    ],
)
def test_ranks_reject_bad_input(scores, columns, message):
    with pytest.raises(DataError, match=message):
        ranks_of_relevant(scores, columns)


def test_metrics_reject_bad_ranks():
    with pytest.raises(DataError, match="1-based"):
        retrieval_metrics(np.array([0, 1]))
    with pytest.raises(DataError, match="no ranks"):
        retrieval_metrics(np.array([]))
