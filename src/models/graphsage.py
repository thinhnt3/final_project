"""GraphSAGE encoder that turns an image's sub-graph into one vector in the CLIP space."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv

from src.core.exceptions import ConfigError, DataError
from src.models.ablations import Ablation
from src.models.graph_data import NodeKind


class ImageNodeFeature(StrEnum):
    """Initial feature of the image node."""

    LEARNED = "learned"        # one trainable vector shared by all images: the output comes from the graph only
    CLIP_IMAGE = "clip_image"  # the image's own frozen CLIP vector


@dataclass(frozen=True)
class GraphSageConfig:
    """Architecture of the graph encoder."""

    hidden_dim: int
    num_layers: int
    directed: bool
    image_node: ImageNodeFeature
    ablation: Ablation = Ablation.FULL
    ablation_seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ConfigError("graphsage.hidden_dim and graphsage.num_layers must be >= 1")

    def run_name(self) -> str:
        """Name of training runs with this architecture, e.g. `directed-learned-l3-h512`."""
        direction = "directed" if self.directed else "undirected"
        base = f"{direction}-{self.image_node.value}-l{self.num_layers}-h{self.hidden_dim}"
        return base if self.ablation is Ablation.FULL else f"{base}-{self.ablation.value}-a{self.ablation_seed}"


class DirectedSAGELayer(nn.Module):
    """GraphSAGE layer that keeps edge direction: one SAGEConv per direction, summed."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.forward_conv = SAGEConv(in_dim, out_dim, aggr="mean")   # messages along the edge
        self.backward_conv = SAGEConv(in_dim, out_dim, aggr="mean")  # messages against the edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Sum of the mean aggregation over incoming and over outgoing neighbours."""
        result: torch.Tensor = self.forward_conv(x, edge_index) + self.backward_conv(x, edge_index.flip(0))
        return result


class UndirectedSAGELayer(nn.Module):
    """Plain GraphSAGE layer: one SAGEConv over the edges in both directions."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean aggregation over all neighbours, whatever the edge direction."""
        result: torch.Tensor = self.conv(x, torch.cat([edge_index, edge_index.flip(0)], dim=1))
        return result


class GraphImageEncoder(nn.Module):
    """Sub-graph of an image → unit vector comparable with CLIP text vectors.

    Object and relation nodes start from frozen CLIP text vectors of their concept and
    predicate. After the GraphSAGE layers the image node's state is projected and normalized.
    Only the layers, the projection and the learned image vector are trained.
    """

    concept_vectors: torch.Tensor
    predicate_vectors: torch.Tensor
    image_vectors: torch.Tensor

    def __init__(
        self,
        config: GraphSageConfig,
        concept_vectors: torch.Tensor,
        predicate_vectors: torch.Tensor,
        image_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        dim = concept_vectors.shape[1]
        if predicate_vectors.shape[1] != dim:
            raise DataError("concept and predicate vectors differ in size")
        self.config = config
        # Frozen lookup tables; not saved in checkpoints, they are restored from the embedding store.
        self.register_buffer("concept_vectors", concept_vectors.float(), persistent=False)
        self.register_buffer("predicate_vectors", predicate_vectors.float(), persistent=False)
        if config.image_node is ImageNodeFeature.CLIP_IMAGE:
            if image_vectors is None or image_vectors.shape[1] != dim:
                raise DataError("image vectors of the CLIP size are required when the image node carries CLIP")
            self.register_buffer("image_vectors", image_vectors.float(), persistent=False)
        else:
            # Same scale as a unit vector spread over `dim` components.
            self.image_token = nn.Parameter(torch.randn(dim) / dim**0.5)

        # Direction ablation reuses the existing undirected layer.
        use_direction = config.directed and config.ablation is not Ablation.UNDIRECTED
        layer = DirectedSAGELayer if use_direction else UndirectedSAGELayer
        sizes = [dim] + [config.hidden_dim] * config.num_layers
        self.layers = nn.ModuleList(layer(sizes[i], sizes[i + 1]) for i in range(config.num_layers))
        self.project = nn.Linear(config.hidden_dim, dim)

    def node_features(self, batch: Batch) -> torch.Tensor:
        """Initial feature of every node, read from the table of its kind."""
        kind, row = batch.node_kind, batch.feature_row
        features = torch.zeros(kind.shape[0], self.concept_vectors.shape[1], device=row.device)
        is_object, is_relation, is_image = (kind == k for k in (NodeKind.OBJECT, NodeKind.RELATION, NodeKind.IMAGE))
        features[is_object] = self.concept_vectors[row[is_object]]

        # Mask predicate features while retaining relation nodes and topology.
        if self.config.ablation is not Ablation.DROP_PREDICATES:
            features[is_relation] = self.predicate_vectors[row[is_relation]]
        if self.config.image_node is ImageNodeFeature.CLIP_IMAGE:
            features[is_image] = self.image_vectors[row[is_image]]
        else:
            features[is_image] = self.image_token.to(features.dtype)
        return features

    def forward(self, batch: Batch) -> torch.Tensor:
        """Unit-length graph vector of each image in the batch, shape (graphs, clip dim)."""
        x = self.node_features(batch)
        for position, layer in enumerate(self.layers):
            x = layer(x, batch.edge_index)
            # No ReLU after the last layer: CLIP vectors have negative components, so must these.
            if position < len(self.layers) - 1:
                x = torch.relu(x)
        image_nodes = batch.ptr[:-1]  # the image is the first node of every sub-graph
        return nn.functional.normalize(self.project(x[image_nodes]), dim=-1)
