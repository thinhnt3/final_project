import pytest
import torch

from src.core.exceptions import DataError
from src.models.explain import explain_graph_score
from src.models.graph_data import NodeKind
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature

DIM = 8
CONCEPTS = {0: "man", 1: "bike", 2: "tree"}
PREDICATES = {0: "riding", 1: "near"}
# man riding bike, man near tree
GRAPH = {"vg_image_id": 7, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2]]}


class _SumOfNodes(torch.nn.Module):
    """Stand-in encoder: graph vector = normalized sum of one basis vector per node.

    Concept c contributes axis c, predicate p contributes axis 4 + p, so the effect of
    removing a relation is known exactly.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        axis = torch.where(batch.node_kind == NodeKind.RELATION, batch.feature_row + 4, batch.feature_row)
        features = torch.nn.functional.one_hot(axis, DIM).float()
        features[batch.node_kind == NodeKind.IMAGE] = 0.0
        sums = torch.zeros(batch.num_graphs, DIM).index_add_(0, batch.batch, features)
        return torch.nn.functional.normalize(sums, dim=-1)


def _axis(index):
    return torch.nn.functional.one_hot(torch.tensor(index), DIM).float()


def test_relation_matching_the_query_contributes_most():
    riding_query = _axis(4)  # the axis of predicate `riding`
    score, contributions = explain_graph_score(_SumOfNodes(), GRAPH, 0, riding_query, CONCEPTS, PREDICATES)
    assert score == pytest.approx(1 / 5**0.5)  # five unit axes: three concepts, two predicates
    assert [(c.subject, c.predicate, c.object) for c in contributions] == [
        ("man", "riding", "bike"), ("man", "near", "tree"),
    ]
    # Without `riding` the query axis is gone: the whole score is lost.
    assert contributions[0].score_drop == pytest.approx(score)
    # Without `near` the vector has four axes left, so the score goes up: a negative drop.
    assert contributions[1].score_drop == pytest.approx(score - 1 / 4**0.5)
    assert contributions[0].score_drop > 0 > contributions[1].score_drop


def test_the_other_query_reverses_the_ranking():
    _, contributions = explain_graph_score(_SumOfNodes(), GRAPH, 0, _axis(5), CONCEPTS, PREDICATES)
    assert [c.predicate for c in contributions] == ["near", "riding"]


def test_image_without_relations_has_a_score_and_no_contributions():
    graph = {"vg_image_id": 1, "concept_ids": [0, 1], "triplets": []}
    score, contributions = explain_graph_score(_SumOfNodes(), graph, 0, _axis(0), CONCEPTS, PREDICATES)
    assert score == pytest.approx(1 / 2**0.5) and contributions == []


def test_works_with_the_real_encoder_and_leaves_it_in_eval_mode():
    torch.manual_seed(0)
    unit = lambda rows: torch.nn.functional.normalize(torch.randn(rows, DIM), dim=-1)  # noqa: E731
    config = GraphSageConfig(hidden_dim=8, num_layers=3, directed=True, image_node=ImageNodeFeature.LEARNED)
    model = GraphImageEncoder(config, unit(3), unit(2)).train()
    score, contributions = explain_graph_score(model, GRAPH, 0, unit(1)[0], CONCEPTS, PREDICATES)
    assert -1.0 <= score <= 1.0 and len(contributions) == 2
    assert contributions[0].score_drop >= contributions[1].score_drop
    assert not model.training
    assert contributions[0].to_dict().keys() == {"subject", "predicate", "object", "score_drop"}


def test_query_must_be_a_single_vector():
    with pytest.raises(DataError, match="one-dimensional"):
        explain_graph_score(_SumOfNodes(), GRAPH, 0, torch.zeros(2, DIM), CONCEPTS, PREDICATES)
