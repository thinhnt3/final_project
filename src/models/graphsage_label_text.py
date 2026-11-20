"""GraphSAGE encoder plus a separate branch for the CLIP text of the object labels.

Control for AdaMF-MAT: same GraphSAGE as the `clip_image` baseline (the image node starts
from the image's CLIP vector, read out at the image node), plus the label-text modality of
AdaMF-MAT built the same way, concatenated before the output projection. No fusion weights,
no generator: if this closes the gap to AdaMF-MAT, the gap comes from the label-text branch.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch

from src.models.adamf_mat import label_text_vectors
from src.models.graphsage import GraphImageEncoder, GraphSageConfig


class LabelTextGraphEncoder(nn.Module):
    """Sub-graph of an image → unit vector comparable with CLIP text vectors."""

    concept_vectors: torch.Tensor

    def __init__(
        self,
        config: GraphSageConfig,
        text_dim: int,
        concept_vectors: torch.Tensor,
        predicate_vectors: torch.Tensor,
        image_vectors: torch.Tensor,
    ) -> None:
        super().__init__()
        clip_dim = concept_vectors.shape[1]
        self.register_buffer("concept_vectors", concept_vectors.float(), persistent=False)
        self.graph = GraphImageEncoder(config, concept_vectors, predicate_vectors, image_vectors)
        self.text_proj = nn.Linear(clip_dim, text_dim)
        self.project = nn.Linear(config.hidden_dim + text_dim, clip_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Unit-length vector of each image in the batch, shape (graphs, clip dim)."""
        graph = self.graph.image_node_state(batch)
        text = self.text_proj(label_text_vectors(self.concept_vectors, batch))
        return nn.functional.normalize(self.project(torch.cat([graph, text], dim=-1)), dim=-1)
