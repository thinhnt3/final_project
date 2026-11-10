"""Minimal GraphSAGE encoder used as the replaceable learned graph backend."""
from __future__ import annotations

import torch
from torch import nn


class MeanGraphSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply mean-neighbor aggregation; edge_index is [2, E] source -> target."""
        source, target = edge_index
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, target, x[source])
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        neighbor_mean = neighbor_sum / degree.clamp_min(1).unsqueeze(1)
        return torch.relu(self.self_linear(x) + self.neighbor_linear(neighbor_mean))


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 128, out_features: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            MeanGraphSAGELayer(in_features, hidden_features),
            MeanGraphSAGELayer(hidden_features, out_features),
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return torch.nn.functional.normalize(x, dim=-1)


class GraphSAGEGraphScorer(nn.Module):
    """Trainable graph-to-query scorer using frozen text embeddings as node features."""

    def __init__(self, feature_dim: int = 512, hidden_features: int = 256):
        super().__init__()
        self.encoder = GraphSAGEEncoder(feature_dim, hidden_features, feature_dim)

    def encode_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_features, edge_index).mean(dim=0)

    def score(self, query: torch.Tensor, graph_embedding: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(query.unsqueeze(0), graph_embedding.unsqueeze(0)).squeeze(0)


def graph_inputs(graph: dict, label_vectors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = list(graph["objects"].values())
    node_ids = {label: i for i, label in enumerate(labels)}
    features = torch.stack([label_vectors[label] for label in labels])
    edges = []
    for relation in graph["relationships"]:
        source = node_ids.get(graph["objects"].get(relation["subject_id"]))
        target = node_ids.get(graph["objects"].get(relation["object_id"]))
        if source is not None and target is not None:
            edges.extend([(source, target), (target, source)])
    if not edges:
        edges = [(0, 0)]
    return features, torch.tensor(edges, dtype=torch.long).t().contiguous()
