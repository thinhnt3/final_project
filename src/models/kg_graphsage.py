"""GraphSAGE over a knowledge graph: every entity becomes one vector in the CLIP space."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.core.exceptions import ConfigError, DataError


@dataclass(frozen=True)
class KgSageConfig:
    """Architecture of the entity encoder. One layer reaches one hop further in the graph."""

    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ConfigError("kg_graphsage.hidden_dim and kg_graphsage.num_layers must be >= 1")


def mean_at(messages: torch.Tensor, targets: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Mean of the messages arriving at each node; zeros for a node that receives none."""
    total = torch.zeros(n_nodes, messages.shape[1], dtype=messages.dtype, device=messages.device)
    total.index_add_(0, targets, messages)
    count = torch.zeros(n_nodes, dtype=messages.dtype, device=messages.device)
    count.index_add_(0, targets, torch.ones_like(targets, dtype=messages.dtype))
    return total / count.clamp(min=1).unsqueeze(1)


class RelationSAGELayer(nn.Module):
    """GraphSAGE mean aggregation where each message also carries its relation.

    A neighbour sends `its state + a learned vector of the relation`. Messages along the
    edge (head → tail) and against it (tail → head) are averaged separately and get their
    own weights, so "X directed Y" and "Y directed X" differ. The node's own state goes
    through a third weight, as in plain GraphSAGE.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int) -> None:
        super().__init__()
        self.own = nn.Linear(in_dim, out_dim)
        self.along = nn.Linear(in_dim, out_dim, bias=False)
        self.against = nn.Linear(in_dim, out_dim, bias=False)
        self.relation = nn.Embedding(n_relations, in_dim)
        nn.init.normal_(self.relation.weight, std=in_dim**-0.5)

    def forward(self, x: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        """New node states from `triples`, a (triples × 3) tensor of head, relation, tail ids."""
        heads, relations, tails = triples.unbind(dim=1)
        relation_states = self.relation(relations)
        to_tail = mean_at(x[heads] + relation_states, tails, x.shape[0])
        to_head = mean_at(x[tails] + relation_states, heads, x.shape[0])
        result: torch.Tensor = self.own(x) + self.along(to_tail) + self.against(to_head)
        return result


class KgEntityEncoder(nn.Module):
    """Entity features + graph → one unit vector per entity, comparable with CLIP text vectors.

    Entities start from frozen CLIP text vectors of their names. Only the layers and the
    final projection are trained.
    """

    entity_features: torch.Tensor

    def __init__(self, config: KgSageConfig, entity_features: torch.Tensor, n_relations: int) -> None:
        super().__init__()
        if entity_features.ndim != 2 or n_relations < 1:
            raise DataError("entity features must be a matrix and the graph needs at least one relation")
        dim = entity_features.shape[1]
        self.config = config
        # Frozen lookup table; not saved in checkpoints, it is restored from the embedding store.
        self.register_buffer("entity_features", entity_features.float(), persistent=False)
        sizes = [dim] + [config.hidden_dim] * config.num_layers
        self.layers = nn.ModuleList(
            RelationSAGELayer(sizes[i], sizes[i + 1], n_relations) for i in range(config.num_layers)
        )
        self.project = nn.Linear(config.hidden_dim, dim)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Unit vector of every entity, shape (entities, clip dim), using only the given triples as edges."""
        x = self.entity_features
        for position, layer in enumerate(self.layers):
            x = layer(x, triples)
            # No ReLU after the last layer: CLIP vectors have negative components, so must these.
            if position < len(self.layers) - 1:
                x = torch.relu(x)
        return nn.functional.normalize(self.project(x), dim=-1)
