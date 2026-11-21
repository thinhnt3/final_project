import numpy as np
import pytest

from src.core.exceptions import DataError
from src.evaluation.entity_retrieval import entity_scores, filtered_ranks


def test_entity_scores_take_the_best_image():
    image_scores = np.array([[0.1, 0.9, 0.2, 0.3, 0.8, 0.4]], dtype=np.float32)
    # entity 0 owns columns 0-1, entity 1 column 2, entity 2 columns 3-5
    assert entity_scores(image_scores, np.array([0, 2, 3])).tolist() == [pytest.approx([0.9, 0.2, 0.8])]
    with pytest.raises(DataError, match="group starts"):
        entity_scores(image_scores, np.array([1, 2]))


def test_filtered_ranks_skip_other_correct_entities():
    scores = np.array([[0.9, 0.8, 0.7, 0.6]])
    # Columns 0 and 2 are both relevant; column 1 is a wrong entity.
    ranks, rows = filtered_ranks(scores, [[0, 2]], [[]])
    assert ranks.tolist() == [1, 2] and rows.tolist() == [0, 0]
    # Column 1 is a known answer from another split: it no longer outranks column 2.
    ranks, _ = filtered_ranks(scores, [[2]], [[0, 1]])
    assert ranks.tolist() == [1]


def test_ties_count_against_the_relevant_entity():
    ranks, _ = filtered_ranks(np.array([[0.5, 0.5, 0.5]]), [[1]], [[]])
    assert ranks.tolist() == [3]


def test_bad_inputs_fail():
    with pytest.raises(DataError, match="no relevant"):
        filtered_ranks(np.zeros((1, 3)), [[]], [[]])
    with pytest.raises(DataError, match="outside"):
        filtered_ranks(np.zeros((1, 3)), [[5]], [[]])
