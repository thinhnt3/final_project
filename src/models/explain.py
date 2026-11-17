"""Explain a graph score: how much each relation of the image contributes to it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from src.core.exceptions import DataError
from src.models.graph_data import image_graph_to_data


@dataclass(frozen=True)
class RelationContribution:
    """One relation of the image and the graph-score change when it is removed."""

    subject: str
    predicate: str
    object: str
    score_drop: float  # graph score with all relations minus the score without this one

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return vars(self).copy()


@torch.no_grad()
def explain_graph_score(
    model: nn.Module,
    image_graph: Mapping[str, Any],
    image_row: int,
    query_vector: torch.Tensor,
    concept_text: Mapping[int, str],
    predicate_text: Mapping[int, str],
) -> tuple[float, list[RelationContribution]]:
    """Graph score of one image for a query, and its relations ranked by contribution.

    Each relation is removed in turn (its two objects stay) and the score is recomputed. A
    large positive drop means the score relied on that relation; a negative one means the
    relation pulled the score down. The same removal is what a fidelity test perturbs.
    """
    if query_vector.ndim != 1:
        raise DataError(f"query vector must be one-dimensional, got shape {tuple(query_vector.shape)}")
    triplets = [tuple(int(v) for v in t) for t in image_graph["triplets"]]
    variants = [image_graph] + [
        {**image_graph, "triplets": triplets[:position] + triplets[position + 1:]}
        for position in range(len(triplets))
    ]
    model.eval()
    device = next(model.parameters()).device
    batch = Batch.from_data_list([image_graph_to_data(variant, image_row) for variant in variants]).to(device)
    scores = (model(batch) @ query_vector.to(device)).cpu()
    full_score = float(scores[0])
    contributions = [
        RelationContribution(
            subject=concept_text[subject],
            predicate=predicate_text[predicate],
            object=concept_text[target],
            score_drop=full_score - float(scores[1 + position]),
        )
        for position, (subject, predicate, target) in enumerate(triplets)
    ]
    # Largest contribution first; equal drops keep the order of the image graph.
    return full_score, sorted(contributions, key=lambda c: -c.score_drop)
