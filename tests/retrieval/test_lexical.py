import math

import numpy as np
import pytest

from src.core.exceptions import DataError
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData
from src.retrieval.lexical import (
    graph_stems,
    inverse_document_frequency,
    lexical_graph_scores,
    lexical_scores,
    stem,
    text_stems,
)

CONCEPTS = {0: "man", 1: "bikes", 2: "wine glass"}
PREDICATES = {0: "riding", 1: "next to"}
GRAPH = {"concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2]]}


@pytest.mark.parametrize(
    ("word", "expected"),
    [("bikes", "bike"), ("bike", "bike"), ("glass", "glass"), ("bus", "bus"), ("gas", "gas"), ("cars", "car")],
)
def test_stem_drops_one_plural_s_only(word, expected):
    assert stem(word) == expected


def test_text_stems_drop_function_words_and_plurals():
    assert text_stems("Two men riding bikes on the street.") == {"men", "riding", "bike", "street"}


def test_graph_stems_with_and_without_predicates():
    objects_only = graph_stems(GRAPH, CONCEPTS, PREDICATES, use_predicates=False)
    assert objects_only == {"man", "bike", "wine", "glass"}
    with_predicates = graph_stems(GRAPH, CONCEPTS, PREDICATES, use_predicates=True)
    assert with_predicates == objects_only | {"riding", "next", "to"}


def test_idf_is_higher_for_rarer_stems_and_highest_for_unseen():
    idf, unseen = inverse_document_frequency([{"man", "bike"}, {"man"}, {"man", "dog"}])
    assert idf["man"] == pytest.approx(math.log(1 + 3 / 4))
    assert idf["bike"] == idf["dog"] == pytest.approx(math.log(1 + 3 / 2))
    assert idf["man"] < idf["bike"] < unseen == pytest.approx(math.log(4))
    with pytest.raises(DataError, match="zero train"):
        inverse_document_frequency([])


def test_scores_are_the_idf_weight_share_of_matched_caption_stems():
    idf = {"man": 1.0, "bike": 3.0, "dog": 2.0}
    gallery = [{"man", "bike"}, {"man", "dog"}, {"tree"}]
    scores = lexical_scores(["a man riding bikes", "the of a"], gallery, idf, unseen_idf=4.0)
    # Caption stems: man (1), riding (unseen, 4), bike (3) → total 8.
    assert scores[0].tolist() == pytest.approx([4 / 8, 1 / 8, 0.0])
    assert scores[1].tolist() == [0.0, 0.0, 0.0]  # no content words
    assert scores.shape == (2, 3) and scores.min() >= 0.0 and scores.max() <= 1.0


def test_repeated_caption_words_count_once_and_full_match_scores_one():
    scores = lexical_scores(["man man bike"], [{"man", "bike"}], {"man": 1.0, "bike": 3.0}, unseen_idf=4.0)
    assert scores.tolist() == [[1.0]]


def test_rare_match_outranks_common_match():
    idf = {"man": 0.1, "zebra": 5.0}
    scores = lexical_scores(["man and zebra"], [{"man"}, {"zebra"}], idf, unseen_idf=6.0)
    assert scores[0, 1] > scores[0, 0]


def test_empty_inputs_are_rejected():
    with pytest.raises(DataError, match="at least one"):
        lexical_scores([], [{"man"}], {}, unseen_idf=1.0)
    with pytest.raises(DataError, match="at least one"):
        lexical_scores(["man"], [], {}, unseen_idf=1.0)


def test_gallery_image_without_words_scores_zero():
    scores = lexical_scores(["a man"], [set(), {"man"}], {"man": 1.0}, unseen_idf=2.0)
    assert scores.tolist() == [[0.0, 1.0]]
    assert np.isfinite(scores).all()


def _split_data(split, image_ids, query_ids):
    empty = np.zeros((0, 0), dtype=np.float32)
    return SplitData(
        split=split, image_ids=tuple(image_ids), query_ids=tuple(query_ids),
        relevant_columns=np.zeros(len(query_ids), dtype=int), image_vectors=empty, query_vectors=empty,
    )


TEXT = {0: "man", 1: "bike", 2: "zebra", 3: "dog"}
PRED = {0: "riding"}
# Train: `man` in both images, `bike` in one. Val/test bring `zebra` and `dog` many times.
GRAPHS = [
    {"vg_image_id": 1, "split": "train", "concept_ids": [0, 1], "triplets": [[0, 0, 1]]},
    {"vg_image_id": 2, "split": "train", "concept_ids": [0], "triplets": []},
    {"vg_image_id": 10, "split": "val", "concept_ids": [0, 2], "triplets": []},
    {"vg_image_id": 11, "split": "val", "concept_ids": [3, 1], "triplets": [[3, 0, 1]]},
    {"vg_image_id": 20, "split": "test", "concept_ids": [2], "triplets": []},
]
QUERY_TEXT = {"q-man": "a man", "q-bike": "a dog riding a bike", "q-zebra": "zebra"}


def _scores(use_predicates=True, graphs=GRAPHS, val_images=(10, 11), val_queries=("q-man", "q-bike")):
    data = {Split.VAL: _split_data(Split.VAL, val_images, val_queries)}
    return lexical_graph_scores(graphs, TEXT, PRED, QUERY_TEXT, data, use_predicates=use_predicates)[Split.VAL]


def test_graph_scores_follow_the_id_order_of_the_split_not_of_the_files():
    forward = _scores()
    assert forward.shape == (2, 2)
    assert forward[0, 0] == 1.0 and forward[0, 1] == 0.0      # "a man" matches image 10 only
    assert forward[1, 1] == 1.0 and forward[1, 0] == 0.0      # dog, riding, bike all in image 11
    swapped = _scores(val_images=(11, 10), val_queries=("q-bike", "q-man"))
    assert np.array_equal(swapped, forward[::-1, ::-1])
    assert np.array_equal(_scores(graphs=GRAPHS[::-1]), forward)


def test_predicate_words_count_only_when_requested():
    with_predicates = _scores(use_predicates=True)
    objects_only = _scores(use_predicates=False)
    assert with_predicates[1, 1] == 1.0
    assert objects_only[1, 1] < 1.0  # "riding" is no longer in the graph words


def test_idf_ignores_validation_and_test_graphs():
    # `zebra` and `dog` never occur in train, so they must weigh as unseen stems however
    # often they occur in validation and test.
    crowded = GRAPHS + [
        {"vg_image_id": 100 + i, "split": split, "concept_ids": [2, 3], "triplets": []}
        for i, split in enumerate(["val", "test"] * 20)
    ]
    assert np.array_equal(_scores(graphs=crowded), _scores())
    unseen = math.log(1 + 2)                # two train images
    bike, riding = math.log(1 + 2 / 2), math.log(1 + 2 / 2)
    # "a dog riding a bike" against image 10 (man, zebra): nothing matches; against a graph
    # holding only `bike`, the share is bike / (dog + riding + bike) with dog unseen.
    only_bike = [*GRAPHS, {"vg_image_id": 12, "split": "val", "concept_ids": [1], "triplets": []}]
    scores = _scores(graphs=only_bike, val_images=(12,), val_queries=("q-bike",))
    assert scores[0, 0] == pytest.approx(bike / (unseen + riding + bike))


def test_missing_graph_or_query_text_is_rejected():
    with pytest.raises(DataError, match="1 images without a graph"):
        _scores(val_images=(10, 999))
    with pytest.raises(DataError, match="1 queries without text"):
        _scores(val_queries=("q-man", "q-none"))
