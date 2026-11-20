"""Relation checks run on validation after every epoch: does the encoder read the scene-graph edges?

Uses the validation relation queries (`scripts.build_relation_queries --split val`). For
each query the relevant image is ranked against its confusers (images with the same two
objects but not the triplet), first with its full graph, then with the queried edge
removed. `edge_sensitivity` is the accuracy lost to that removal: near zero means the
encoder ignores the edges, whatever its caption metrics say.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from src.core.exceptions import DataError
from src.evaluation.relation_queries import QueryKind, confuser_ranks, contrast_confusers
from src.models.ablations import remove_triplet
from src.models.graph_data import image_graph_to_data
from src.models.train import TrainingConfig, encode_graphs


class RelationValidator:
    """Relation accuracy, contrast accuracy and edge sensitivity of an encoder on fixed queries."""

    def __init__(
        self,
        queries: Sequence[Mapping[str, Any]],
        query_vectors: np.ndarray,
        image_graphs: Mapping[int, Mapping[str, Any]],
        image_row: Mapping[int | str, int],
        config: TrainingConfig,
    ) -> None:
        relation = [q for q in queries if q["kind"] == QueryKind.RELATION.value]
        if not relation or len(relation) != len(query_vectors):
            raise DataError(f"{len(relation)} relation queries but {len(query_vectors)} query vectors")
        image_ids = sorted({int(i) for q in relation for i in (*q["relevant_ids"], *q["confuser_ids"])})
        column = {image_id: c for c, image_id in enumerate(image_ids)}
        self.config = config
        self.texts = torch.from_numpy(np.asarray(query_vectors, dtype=np.float32))
        self.graphs = [image_graph_to_data(image_graphs[i], image_row[i]) for i in image_ids]
        self.relevant = [column[int(q["relevant_ids"][0])] for q in relation]
        self.confusers = [[column[int(i)] for i in q["confuser_ids"]] for q in relation]
        self.contrast = [[column[i] for i in contrast_confusers(q, image_graphs)] for q in relation]
        self.masked_graphs = [
            image_graph_to_data(remove_triplet(image_graphs[int(q["relevant_ids"][0])], q["triplet"]),
                                image_row[int(q["relevant_ids"][0])])
            for q in relation
        ]

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> dict[str, float]:
        """Metrics of the model as it is now; leaves it in eval mode, as `evaluate` does."""
        vectors = encode_graphs(model, self.graphs, self.config)
        masked = encode_graphs(model, self.masked_graphs, self.config)
        scores = (self.texts @ vectors.T).numpy()
        rows = np.arange(len(self.relevant))
        masked_scores = scores.copy()
        masked_scores[rows, self.relevant] = (self.texts * masked).sum(dim=1).numpy()
        accuracy = float((confuser_ranks(scores, self.relevant, self.confusers) == 1).mean())
        masked_accuracy = float((confuser_ranks(masked_scores, self.relevant, self.confusers) == 1).mean())
        with_contrast = [r for r in rows if self.contrast[r]]
        contrast = confuser_ranks(
            scores[with_contrast], [self.relevant[r] for r in with_contrast], [self.contrast[r] for r in with_contrast]
        )
        return {
            "relation_acc": accuracy,
            "relation_contrast_acc": float((contrast == 1).mean()),
            "edge_sensitivity": accuracy - masked_accuracy,
        }
