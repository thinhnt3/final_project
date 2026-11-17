from copy import deepcopy

import pytest
import torch

from src.core.exceptions import ConfigError, DataError
from src.evaluation.fidelity import evaluate_fidelity
from src.models.graph_data import NodeKind
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature

GRAPH = {"vg_image_id": 7, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2]]}
DIM = 8


class SumOfNodes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        axes = torch.where(batch.node_kind == NodeKind.RELATION, batch.feature_row + 4, batch.feature_row)
        features = torch.nn.functional.one_hot(axes, DIM).float()
        features[batch.node_kind == NodeKind.IMAGE] = 0
        sums = torch.zeros(batch.num_graphs, DIM).index_add_(0, batch.batch, features)
        return torch.nn.functional.normalize(sums, dim=-1)


def axis(index):
    return torch.nn.functional.one_hot(torch.tensor(index), DIM).float()


def test_known_support_beats_disjoint_control_and_scales_to_final_score():
    original = deepcopy(GRAPH)
    result = evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(4), budget=3, seed=13, alpha=0.8)
    assert result.selected_indices == (0,) and result.control_indices == (1,)
    assert result.relation_count == 2 and result.requested_budget == 3
    assert result.graph_score == pytest.approx(1 / 5**0.5)
    assert result.removed_graph_score == 0
    assert result.control_graph_score == pytest.approx(0.5)
    values = result.to_dict()
    assert values["final_score_drop"] == pytest.approx(0.2 / 5**0.5)
    assert values["control_final_score_drop"] < 0 < values["final_score_drop"]
    assert values["sparsity"] == 0.5 and result.status == "controlled"
    assert original == GRAPH
    assert result == evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(4), budget=3, seed=13, alpha=0.8)


def test_no_support_or_no_disjoint_control_is_explicit():
    result = evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(0), budget=1, seed=13, alpha=0.8)
    assert result.status == "no_supporting_relations"
    assert result.to_dict()["control_final_score_drop"] is None
    for triplets in ([], [[0, 0, 1]]):
        result = evaluate_fidelity(
            SumOfNodes(), {**GRAPH, "triplets": triplets}, 0, axis(4), budget=1, seed=13, alpha=0.8
        )
        assert result.status == "insufficient_relations" and not result.selected_indices


def test_joint_effect_is_recomputed_instead_of_summing_individual_deltas():
    graph = {**GRAPH, "concept_ids": [0, 1], "triplets": [[0, p, 1] for p in range(4)]}
    query = (axis(4) + axis(5)) / 2**0.5
    result = evaluate_fidelity(SumOfNodes(), graph, 0, query, budget=2, seed=37, alpha=0)
    assert len(result.selected_indices) == len(result.control_indices) == 2
    assert not set(result.selected_indices) & set(result.control_indices)
    assert result.removed_graph_score == 0
    assert result.to_dict()["final_score_drop"] == pytest.approx(2 / 12**0.5)


def test_alpha_one_has_zero_fused_drop_even_when_graph_changes():
    result = evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(4), budget=1, seed=13, alpha=1)
    assert result.to_dict()["final_score_drop"] == result.to_dict()["control_final_score_drop"] == 0


@pytest.mark.parametrize("options", [{"budget": 0}, {"seed": -1}, {"alpha": float("nan")}, {"alpha": 2}])
def test_invalid_protocol_is_rejected(options):
    with pytest.raises(ConfigError):
        evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(4), **({"budget": 1, "seed": 13, "alpha": 0.8} | options))


def test_bad_query_is_rejected():
    with pytest.raises(DataError):
        evaluate_fidelity(SumOfNodes(), GRAPH, 0, torch.zeros(2, DIM), budget=1, seed=13, alpha=0.8)


def test_existing_graphsage_accepts_perturbed_graphs():
    torch.manual_seed(13)
    vectors = lambda n: torch.nn.functional.normalize(torch.randn(n, DIM), dim=-1)  # noqa: E731
    model = GraphImageEncoder(GraphSageConfig(8, 3, True, ImageNodeFeature.LEARNED), vectors(3), vectors(2))
    result = evaluate_fidelity(model, GRAPH, 0, vectors(1)[0], budget=1, seed=13, alpha=0.8)
    assert -1 <= result.graph_score <= 1 and not model.training


def test_non_candidate_graph_changes_do_not_change_final_score():
    result = evaluate_fidelity(SumOfNodes(), GRAPH, 0, axis(4), budget=1, seed=13, alpha=0.8, is_candidate=False)
    assert result.graph_score > result.removed_graph_score
    assert result.to_dict()["final_score_drop"] == result.to_dict()["control_final_score_drop"] == 0
    assert not result.is_candidate
