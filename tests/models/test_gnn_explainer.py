import numpy as np
import pytest
import torch

from src.core.exceptions import ConfigError, DataError
from src.models.ablations import Ablation
from src.models.gnn_explainer import GnnExplainerConfig, gnn_explainer_relation_scores
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature

DIM = 8
# man riding bike, man near tree, bike near tree
GRAPH = {"vg_image_id": 7, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2], [1, 1, 2]]}
CONFIG = GnnExplainerConfig(epochs=20, learning_rate=0.01)


def _encoder(ablation=Ablation.FULL):
    torch.manual_seed(0)
    config = GraphSageConfig(hidden_dim=16, num_layers=3, directed=True, image_node=ImageNodeFeature.LEARNED,
                             ablation=ablation)
    return GraphImageEncoder(config, torch.randn(3, DIM), torch.randn(2, DIM))


def test_one_importance_per_relation_between_zero_and_one():
    scores = gnn_explainer_relation_scores(_encoder(), GRAPH, 0, torch.randn(DIM), CONFIG, seed=1)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_same_seed_gives_the_same_mask():
    model, query = _encoder(), torch.randn(DIM)
    first = gnn_explainer_relation_scores(model, GRAPH, 0, query, CONFIG, seed=5)
    again = gnn_explainer_relation_scores(model, GRAPH, 0, query, CONFIG, seed=5)
    assert first == pytest.approx(again)


def test_explaining_leaves_the_encoder_unchanged():
    model, query = _encoder(), torch.randn(DIM)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    gnn_explainer_relation_scores(model, GRAPH, 0, query, CONFIG, seed=1)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_undirected_encoder_and_empty_graph_are_rejected():
    query = torch.randn(DIM)
    with pytest.raises(DataError):
        gnn_explainer_relation_scores(_encoder(Ablation.UNDIRECTED), GRAPH, 0, query, CONFIG, seed=1)
    with pytest.raises(DataError):
        gnn_explainer_relation_scores(_encoder(), {**GRAPH, "triplets": []}, 0, query, CONFIG, seed=1)


def test_config_rejects_non_positive_values():
    with pytest.raises(ConfigError):
        GnnExplainerConfig(epochs=0, learning_rate=0.01)
