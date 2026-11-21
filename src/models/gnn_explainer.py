"""Rank an image's relations with GNNExplainer (Ying et al., 2019) instead of single deletions.

GNNExplainer learns a soft mask over the edges of the image's sub-graph so that the masked
graph keeps the graph score while using as few edges as possible. The ranking it gives does
not come from deleting relations, so deletion-based fidelity tests it independently.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer

from src.core.exceptions import ConfigError, DataError
from src.models.graph_data import IMAGE_NODE, image_graph_to_data
from src.models.graphsage import GraphImageEncoder


@dataclass(frozen=True)
class GnnExplainerConfig:
    """Optimization of one edge mask."""

    epochs: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ConfigError("gnn_explainer.epochs must be >= 1 and learning_rate positive")


class _GraphScore(nn.Module):
    """Graph score of one image for one query, as a function of (node features, edges)."""

    query_vector: torch.Tensor

    def __init__(self, encoder: GraphImageEncoder, query_vector: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("query_vector", query_vector.float(), persistent=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Cosine of the image node's graph vector with the query, shape (1, 1)."""
        image_state = self.encoder.message_passing(x, edge_index)[IMAGE_NODE : IMAGE_NODE + 1]
        vector = nn.functional.normalize(self.encoder.project(image_state), dim=-1)
        result: torch.Tensor = (vector @ self.query_vector).unsqueeze(-1)
        return result


def gnn_explainer_relation_scores(
    model: GraphImageEncoder,
    image_graph: Mapping[str, Any],
    image_row: int,
    query_vector: torch.Tensor,
    config: GnnExplainerConfig,
    seed: int,
) -> np.ndarray:
    """Importance of each relation of the image, in the order of `image_graph["triplets"]`.

    A relation is the pair of edges subject → relation node → object; its importance is the
    mean of the two learned edge masks. Edges image → object are masked too but not reported:
    the explanation is over relations, as in the deletion-based one.
    """
    n_relations = len(image_graph["triplets"])
    if query_vector.ndim != 1 or n_relations == 0:
        raise DataError("need a one-dimensional query and at least one relation to explain")
    if not model.uses_edge_direction:
        # The undirected layer doubles the edge list, so one mask entry per edge no longer fits.
        raise DataError("GNNExplainer is wired for the directed encoder only")
    device = next(model.parameters()).device
    data = image_graph_to_data(image_graph, image_row)
    batch = Batch.from_data_list([data]).to(device)
    model.eval()
    with torch.no_grad():
        features = model.node_features(batch)

    torch.manual_seed(seed)  # the mask starts from random values
    explainer = Explainer(
        model=_GraphScore(model, query_vector.to(device)),
        algorithm=GNNExplainer(epochs=config.epochs, lr=config.learning_rate),
        explanation_type="model",
        edge_mask_type="object",
        model_config={"mode": "regression", "task_level": "graph", "return_type": "raw"},
    )
    edge_mask = explainer(features, batch.edge_index).edge_mask.detach().cpu()
    if not torch.isfinite(edge_mask).all():
        raise DataError("GNNExplainer produced a non-finite edge mask")
    # Edge layout from `image_graph_to_data`: image → object edges first, then two per relation.
    first_relation_edge = len(image_graph["concept_ids"])
    relation_edges = edge_mask[first_relation_edge:].reshape(n_relations, 2)
    scores: np.ndarray = relation_edges.mean(dim=1).numpy().astype(float)
    return scores
