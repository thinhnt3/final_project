import numpy as np
import pytest

from src.evaluation.relation_queries import (
    build_relation_queries,
    chance_accuracy,
    confuser_ranks,
)

CONCEPTS = {0: "man", 1: "horse", 2: "dog"}
PREDICATES = {0: "riding", 1: "near"}


def _graph(image_id: int, concepts: list[int], triplets: list[list[int]], split: str = "test") -> dict:
    return {"vg_image_id": image_id, "split": split, "concept_ids": concepts, "triplets": triplets}


def test_relation_query_needs_a_unique_triplet_and_a_confuser() -> None:
    graphs = [
        _graph(1, [0, 1], [[0, 0, 1]]),  # man riding horse: unique, image 2 also has man + horse
        _graph(2, [0, 1], [[0, 1, 1]]),  # man near horse: unique, image 1 is its confuser
        _graph(3, [0, 2], [[0, 1, 2]]),  # man near dog: no other image has man + dog
        _graph(4, [0, 1], [[0, 0, 1]], split="train"),  # train images never count
    ]
    queries = build_relation_queries(graphs, CONCEPTS, PREDICATES, seed=0)
    by_id = {q["query_id"]: q for q in queries}
    assert set(by_id) == {"relation-1", "objects-1", "relation-2", "objects-2"}
    assert by_id["relation-1"]["text"] == "man riding horse"
    assert by_id["objects-1"]["text"] == "man and horse"
    assert by_id["relation-1"]["confuser_ids"] == [2]
    assert by_id["relation-1"]["relevant_ids"] == [1]


def test_shared_triplet_is_not_a_query() -> None:
    graphs = [_graph(1, [0, 1], [[0, 0, 1]]), _graph(2, [0, 1], [[0, 0, 1]])]
    assert build_relation_queries(graphs, CONCEPTS, PREDICATES, seed=0) == []


def test_confuser_ranks_ignore_other_images_and_count_ties_against() -> None:
    scores = np.array([[0.9, 0.5, 0.95, 0.1], [0.4, 0.4, 0.0, 0.0]])
    # Query 0: column 2 outscores the relevant one but is not a confuser.
    ranks = confuser_ranks(scores, [0, 0], [[1, 3], [1]])
    assert ranks.tolist() == [1, 2]
    assert chance_accuracy([[1, 3], [1]]) == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_contrast_confusers_keep_images_joining_the_pair_by_another_relation():
    from src.evaluation.relation_queries import contrast_confusers

    graph_of = {
        2: {"triplets": [[1, 1, 0]]},  # horse near man: the pair, reversed, another relation
        3: {"triplets": [[0, 1, 2]]},  # man near dog: not the pair
    }
    query = {"triplet": [0, 0, 1], "confuser_ids": [2, 3]}
    assert contrast_confusers(query, graph_of) == [2]
