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
    # Share of entities that, at each training step, are encoded from their neighbours alone.
    structure_only_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ConfigError("kg_graphsage.hidden_dim and kg_graphsage.num_layers must be >= 1")
        if not 0.0 <= self.structure_only_rate < 1.0:
            raise ConfigError(f"kg_graphsage.structure_only_rate must lie in [0, 1), got {self.structure_only_rate}")


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

    def forward(self, x: torch.Tensor, triples: torch.Tensor, keep_own: torch.Tensor | None = None) -> torch.Tensor:
        """New node states from `triples`, a (triples × 3) tensor of head, relation, tail ids.

        `keep_own` is 0 for nodes whose own state is left out, so only their neighbours speak.
        """
        heads, relations, tails = triples.unbind(dim=1)
        relation_states = self.relation(relations)
        to_tail = mean_at(x[heads] + relation_states, tails, x.shape[0])
        to_head = mean_at(x[tails] + relation_states, heads, x.shape[0])
        own: torch.Tensor = self.own(x)
        if keep_own is not None:
            own = own * keep_own.unsqueeze(1)
        return own + self.along(to_tail) + self.against(to_head)


class KgEntityEncoder(nn.Module):
    """Entity features + graph → one unit vector per entity, comparable with CLIP text vectors.

    Entities start from frozen CLIP text vectors of their names. Only the layers and the
    final projection are trained.

    With `structure_only_rate > 0`, a random share of entities is encoded without its own
    state at each training step. The name vector alone already answers name queries, and
    that shortcut lets the model ignore the edges; these rows can only be solved through them.
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
        keep_own = None
        if self.training and self.config.structure_only_rate > 0:
            keep_own = (torch.rand(x.shape[0], device=x.device) >= self.config.structure_only_rate).to(x.dtype)
        for position, layer in enumerate(self.layers):
            x = layer(x, triples, keep_own)
            # No ReLU after the last layer: CLIP vectors have negative components, so must these.
            if position < len(self.layers) - 1:
                x = torch.relu(x)
        return nn.functional.normalize(self.project(x), dim=-1)
