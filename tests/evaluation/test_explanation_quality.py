from copy import deepcopy

import pytest

from src.core.exceptions import DataError
from src.evaluation.explanation_quality import (
    keep_only_scores,
    relation_sparsity,
    retain_relations,
    select_review_queries,
    summarize_quality,
    summarize_validity,
)
from tests.evaluation.test_fidelity import GRAPH3
from tests.models.test_explain import _axis, _SumOfNodes


def test_sparsity_counts_explanation_not_its_deleted_complement():
    result = relation_sparsity(10, [1, 6])
    assert result["explanation_fraction"] == pytest.approx(0.2)
    assert result["sparsity"] == pytest.approx(0.8)
    assert relation_sparsity(3, [0, 1, 2])["sparsity"] == 0
    original = deepcopy(GRAPH3)
    retained = retain_relations(GRAPH3, [1])
    assert retained["triplets"] == [GRAPH3["triplets"][1]]
    assert retained["concept_ids"] == GRAPH3["concept_ids"]
    assert original == GRAPH3


@pytest.mark.parametrize("total,selected", [(0, [0]), (3, []), (3, [1, 1]), (3, [-1]), (3, [3]), (3, [True])])
def test_invalid_explanation_positions_fail(total, selected):
    with pytest.raises(DataError):
        relation_sparsity(total, selected)


def test_keep_only_scores_query_relation_against_equal_size_controls():
    budget = {"selected_indices": [0], "random_indices": [[1], [2]]}
    result = keep_only_scores(
        _SumOfNodes(),
        GRAPH3,
        0,
        _axis(4),
        budget,
        batch_size=1,
    )
    # Three orthogonal object features and the retained query predicate: 1/sqrt(4).
    assert result["selected_score"] == pytest.approx(0.5)
    assert result["random_scores"] == [0.0, 0.0]
    with pytest.raises(DataError):
        keep_only_scores(
            _SumOfNodes(),
            GRAPH3,
            0,
            _axis(4),
            {**budget, "random_indices": [[1, 2]]},
        )


def test_sampling_is_seeded_balanced_and_skips_unevaluable_queries():
    queries = [{"query_id": str(i), "status": "evaluated", "predicted_image_is_relevant": i < 5} for i in range(10)]
    queries.append({"query_id": "skipped", "status": "skipped", "predicted_image_is_relevant": True})
    sample = select_review_queries(queries, 2, 123)
    assert sample == select_review_queries(list(reversed(queries)), 2, 123)
    assert len(sample) == 4
    assert sum(q["predicted_image_is_relevant"] for q in sample) == 2
    with pytest.raises(DataError):
        select_review_queries(queries, 6, 123)


CASES = [
    {"case_id": "a", "predicted_image_is_relevant": True, "relations": [{"index": 0}, {"index": 2}]},
    {"case_id": "b", "predicted_image_is_relevant": False, "relations": [{"index": 1}]},
]
REVIEW = {
    "reviewer_type": "ai_assisted",
    "reviewer_id": "test-reviewer",
    "labels": [
        {"case_id": "a", "relation_index": 0, "image_support": "yes", "query_relevance": "yes", "reason": "visible"},
        {
            "case_id": "a",
            "relation_index": 2,
            "image_support": "uncertain",
            "query_relevance": "yes",
            "reason": "occluded",
        },
        {
            "case_id": "b",
            "relation_index": 1,
            "image_support": "uncertain",
            "query_relevance": "no",
            "reason": "unrelated constraint",
        },
    ],
}


def test_validity_unknowns_are_not_dropped_or_counted_as_agreement():
    result = summarize_validity(CASES, REVIEW)
    joint = result["micro"]["joint_validity"]
    assert (joint["yes"], joint["no"], joint["uncertain"]) == (1, 1, 1)
    assert joint["lower_bound"] == pytest.approx(1 / 3)
    assert joint["upper_bound"] == pytest.approx(2 / 3)
    assert joint["yes_among_resolved"] == pytest.approx(0.5)
    assert result["macro_joint_validity_lower_bound"] == pytest.approx(0.25)
    assert result["reviewer_type"] == "ai_assisted"
    assert result["by_prediction_relevance"]["false"]["joint_validity"]["no"] == 1


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "unknown", "bool_index", "bad_label", "no_reason", "no_type"],
)
def test_invalid_or_incomplete_review_is_rejected(failure):
    review = deepcopy(REVIEW)
    if failure == "missing":
        review["labels"].pop()
    elif failure == "duplicate":
        review["labels"].append(review["labels"][0])
    elif failure == "unknown":
        review["labels"][0]["case_id"] = "other"
    elif failure == "bool_index":
        review["labels"][2]["relation_index"] = True
    elif failure == "bad_label":
        review["labels"][0]["image_support"] = "maybe"
    elif failure == "no_reason":
        review["labels"][0]["reason"] = ""
    else:
        review.pop("reviewer_type")
    with pytest.raises(DataError):
        summarize_validity(CASES, review)


def test_summary_means_queries_and_excludes_skips():
    keys = (
        "available_relations",
        "explanation_relations",
        "explanation_fraction",
        "sparsity",
        "deletion_graph_drop",
        "deletion_top1_lost",
        "keep_graph_drop",
        "keep_random_graph_drop",
        "keep_top1_preserved",
        "keep_random_top1_preserved",
        "keep_rank_increase",
    )
    records = [{"budgets": [{"fraction_requested": 0.1, **dict.fromkeys(keys, value)}]} for value in (0.0, 1.0)]
    records.append({"status": "skipped"})
    summary = summarize_quality(records)
    assert summary[0]["evaluated"] == 2
    assert all(summary[0][key] == 0.5 for key in keys)
