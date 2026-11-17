import numpy as np
import pytest
import torch

from src.core.exceptions import DataError
from src.evaluation.fidelity import joint_relation_fidelity, ranking_effects, remove_relations, target_rank
from src.models.explain import score_graph_variants
from tests.models.test_explain import GRAPH, _axis, _SumOfNodes

GRAPH3 = {**GRAPH, "triplets": [[0, 0, 1], [0, 1, 2], [1, 1, 2]]}


def test_removal_keeps_concepts_and_rejects_invalid_indices():
    g = remove_relations(GRAPH3, [0, 2])
    assert g["concept_ids"] == GRAPH3["concept_ids"]
    assert g["triplets"] == [[0, 1, 2]]
    assert len(GRAPH3["triplets"]) == 3
    for bad in ([1, 1], [-1], [3]):
        with pytest.raises(DataError):
            remove_relations(GRAPH3, bad)


def test_chunked_scoring_is_equivalent():
    variants = [GRAPH3, remove_relations(GRAPH3, [0]), remove_relations(GRAPH3, [1, 2])]
    model = _SumOfNodes()
    torch.testing.assert_close(
        score_graph_variants(
            model,
            variants,
            0,
            _axis(4),
            1,
        ),
        score_graph_variants(
            model,
            variants,
            0,
            _axis(4),
            64,
        ),
    )


def test_joint_control_budget_is_fair_and_seeded_with_signed_effects():
    model = _SumOfNodes()
    r = joint_relation_fidelity(
        model,
        GRAPH3,
        0,
        _axis(4),
        [0.1, 0.25],
        20,
        7,
    )
    assert r == joint_relation_fidelity(
        model,
        GRAPH3,
        0,
        _axis(4),
        [0.1, 0.25],
        20,
        7,
    )
    baseline = 1 / 8**0.5  # objects each weight 1, predicate axes weights 1 and 2
    assert r["baseline_graph_score"] == pytest.approx(baseline)
    for b in r["budgets"]:
        assert b["removed"] == 2  # lower bound for joint-deletion, one relation remains
        assert 0 in b["selected_indices"]  # query-matching predicate
        assert b["selected_score"] == pytest.approx(0)
        assert b["selected_drop"] == pytest.approx(baseline)
        assert len(b["random_indices"]) == 20
        assert all(len(set(indices)) == 2 for indices in b["random_indices"])
        assert any(score > baseline for score in b["random_scores"])  # negative drops retained
        assert b["excess_drop"] == pytest.approx(b["selected_drop"] - b["random_drop_mean"])
    assert (
        joint_relation_fidelity(
            model,
            GRAPH,
            0,
            _axis(4),
            [0.25],
            20,
            7,
        )["status"]
        == "skipped"
    )


def test_ranks_use_frozen_competitors_and_conservative_ties():
    scores = np.array([0.7, 0.6, 0.4])
    assert (
        target_rank(
            scores,
            0,
            0.6,
        )
        == 2
    )
    assert (
        target_rank(
            scores,
            0,
            0.3,
        )
        == 3
    )
    # alpha=1 makes prediction completely independent of graph interventions.
    budget = {"selected_score": -1, "random_scores": [-1, 0, 1]}
    out = ranking_effects(
        scores,
        0,
        0.7,
        1,
        budget,
    )
    assert out["selected_rank_increase"] == out["fused_score_drop"] == 0
    assert out["random_top1_lost_rate"] == 0
    assert np.array_equal(scores, [0.7, 0.6, 0.4])
    out = ranking_effects(
        scores,
        0,
        0.7,
        0.5,
        budget,
    )
    assert out["selected_rank"] == 3 and out["selected_top1_lost"]


@pytest.mark.parametrize("fractions,repeats", [([], 1), ([0], 1), ([1], 1), ([0.2], 0)])
def test_invalid_protocol_is_rejected(
    fractions,
    repeats,
):
    with pytest.raises(DataError):
        joint_relation_fidelity(
            _SumOfNodes(),
            GRAPH3,
            0,
            _axis(4),
            fractions,
            repeats,
            7,
        )
