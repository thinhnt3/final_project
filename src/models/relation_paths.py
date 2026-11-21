"""Path scorer with learned relation weights: walks start at the entity a query names."""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.core.exceptions import ConfigError, DataError, TrainingError
from src.evaluation.entity_retrieval import filtered_ranks
from src.evaluation.metrics import retrieval_metrics
from src.models.kg_train import KgQuerySet, _batch_inputs
from src.models.train import SelectionMetric, StopReason, TrainingConfig, selection_key

log = logging.getLogger(__name__)

_EPOCH_SEED_STRIDE = 100_003
_EVAL_BATCH = 512


@dataclass(frozen=True)
class RelationPathConfig:
    """Longest walk, in edges, that the scorer follows from the anchor."""

    max_hops: int

    def __post_init__(self) -> None:
        if self.max_hops < 1:
            raise ConfigError("relation_paths.max_hops must be >= 1")


@dataclass(frozen=True)
class PathQuerySet:
    """Queries that name an entity: `anchors[i]` is the entity id query i starts from."""

    queries: KgQuerySet
    anchors: torch.Tensor

    def __post_init__(self) -> None:
        if self.anchors.shape != (self.queries.query_vectors.shape[0],):
            raise DataError("every path query needs exactly one anchor")


@dataclass(frozen=True)
class PathTrainingResult:
    """Outcome of one run; `state_dict` comes from the best epoch."""

    history: list[dict[str, Any]]
    best_epoch: int
    stop_reason: StopReason
    state_dict: dict[str, torch.Tensor]


class RelationPathScorer(nn.Module):
    """Scores every entity by the weighted walks that reach it from the query's anchor.

    One unit of mass starts on the anchor and moves along edges for `max_hops` steps. A step
    over an edge is multiplied by a weight that depends on the edge's relation, its direction,
    the step number and the query text, so "the director of X" can follow director edges and
    ignore cast edges. Edges are also divided by the square roots of their two degrees, as in
    the training-free path score, so hubs do not dominate. Before training every weight is 1,
    which makes the model that training-free score.
    """

    sources: torch.Tensor
    targets: torch.Tensor
    edge_types: torch.Tensor
    edge_scale: torch.Tensor

    def __init__(self, config: RelationPathConfig, triples: torch.Tensor, n_entities: int, n_relations: int,
                 query_dim: int) -> None:
        super().__init__()
        if triples.ndim != 2 or triples.shape[1] != 3 or triples.shape[0] == 0:
            raise DataError("the path scorer needs a non-empty (triples × 3) graph")
        heads, relations, tails = triples.unbind(dim=1)
        self.config = config
        self.n_entities = n_entities
        self.n_triples = triples.shape[0]
        # Every triple can be walked both ways; the backward step is its own edge type.
        sources, targets = torch.cat([heads, tails]), torch.cat([tails, heads])
        degree = torch.bincount(sources, minlength=n_entities).float()
        self.register_buffer("sources", sources, persistent=False)
        self.register_buffer("targets", targets, persistent=False)
        self.register_buffer("edge_types", torch.cat([relations, relations + n_relations]), persistent=False)
        self.register_buffer("edge_scale", (degree[sources] * degree[targets]).rsqrt(), persistent=False)
        self.step_weights = nn.Linear(query_dim, config.max_hops * 2 * n_relations)
        nn.init.zeros_(self.step_weights.weight)
        nn.init.constant_(self.step_weights.bias, math.log(math.e - 1.0))  # softplus(bias) = 1

    def forward(self, query_vectors: torch.Tensor, anchors: torch.Tensor,
                hidden: torch.Tensor | None = None) -> torch.Tensor:
        """(queries × entities) path mass; the anchor's own column is 0.

        `hidden` is a (pairs × 2) tensor of (query row, triple row): that triple is removed
        from the graph for that query only. Used in training to hide the fact being asked.
        """
        batch = query_vectors.shape[0]
        rows = torch.arange(batch, device=query_vectors.device)
        weights = nn.functional.softplus(self.step_weights(query_vectors)).view(batch, self.config.max_hops, -1)
        scale = self.edge_scale.expand(batch, -1)
        if hidden is not None and len(hidden):
            scale = scale.clone()
            scale[hidden[:, 0], hidden[:, 1]] = 0.0
            scale[hidden[:, 0], hidden[:, 1] + self.n_triples] = 0.0
        not_anchor = torch.ones(batch, self.n_entities, device=query_vectors.device)
        not_anchor[rows, anchors] = 0.0
        state = 1.0 - not_anchor
        total = torch.zeros_like(state)
        for hop in range(self.config.max_hops):
            messages = state[:, self.sources] * weights[:, hop][:, self.edge_types] * scale
            state = torch.zeros_like(state).index_add(1, self.targets, messages)
            total = total + state
        return total * not_anchor


def path_share_loss(scores: torch.Tensor, targets: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Minus log of the share of path mass that lands on the answer, averaged over queries.

    `skip` marks other correct columns; their mass is left out of the total. A query whose
    answer no walk reaches has no gradient and is left out of the mean.
    """
    mass = scores.masked_fill(skip, 0.0)
    answer = mass.gather(1, targets.unsqueeze(1)).squeeze(1)
    reached = answer > 0
    if not reached.any():
        return scores.sum() * 0.0
    return -(answer[reached] / mass.sum(dim=1)[reached]).log().mean()


def gallery_path_scores(model: RelationPathScorer, data: PathQuerySet, gallery: torch.Tensor) -> np.ndarray:
    """(queries × gallery) scores on the full graph, each row divided by its largest value."""
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(data.anchors), _EVAL_BATCH):
            rows = slice(start, start + _EVAL_BATCH)
            scores = model(data.queries.query_vectors[rows], data.anchors[rows])[:, gallery]
            parts.append(scores / scores.amax(dim=1, keepdim=True).clamp(min=1e-30))
    return torch.cat(parts).numpy().astype(np.float32)


def evaluate_paths(model: RelationPathScorer, data: PathQuerySet, gallery: torch.Tensor,
                   ks: Sequence[int]) -> dict[str, float]:
    """Filtered Recall@K and MRR of the path score alone."""
    ranks, _ = filtered_ranks(gallery_path_scores(model, data, gallery), data.queries.relevant, data.queries.known)
    return retrieval_metrics(ranks, sorted({*ks, 1}))


def fit_paths(
    model: RelationPathScorer,
    gallery: torch.Tensor,
    train: PathQuerySet,
    val: PathQuerySet,
    config: TrainingConfig,
    seed: int,
    ks: Sequence[int],
) -> PathTrainingResult:
    """Train the step weights until validation stops improving; return the best epoch's weights.

    Epoch 0 in the history is the untrained model, i.e. the plain path score. It can be the
    best epoch. `gallery` holds the entity ids that can be answers, in gallery-column order.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    n_queries = len(train.anchors)

    def snapshot() -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    metrics = evaluate_paths(model, val, gallery, ks)
    history: list[dict[str, Any]] = [{"epoch": 0, **{f"val_{k}": round(v, 5) for k, v in metrics.items()}}]
    best_key, best_epoch, best_state = selection_key(metrics, config.selection_metric), 0, snapshot()
    log.info("epoch  0 (untrained) val R@1 %.4f MRR %.4f", metrics[SelectionMetric.RECALL_AT_1.value],
             metrics[SelectionMetric.MRR.value])
    stop_reason = StopReason.MAX_EPOCHS
    for epoch in range(1, config.max_epochs + 1):
        if epoch - 1 - best_epoch >= config.patience:
            stop_reason = StopReason.NO_IMPROVEMENT
            break
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed * _EPOCH_SEED_STRIDE + epoch)
        order = torch.randperm(n_queries, generator=generator)
        model.train()
        losses = []
        for start in range(0, n_queries, config.batch_size):
            rows = order[start:start + config.batch_size]
            targets, skip, _ = _batch_inputs(train.queries, rows, len(gallery), model.n_triples, generator)
            hidden = torch.tensor(
                [(position, triple) for position, row in enumerate(rows.tolist())
                 for triple in train.queries.own_triples[row]], dtype=torch.long,
            ).reshape(-1, 2)
            scores = model(train.queries.query_vectors[rows], train.anchors[rows], hidden)[:, gallery]
            loss = path_share_loss(scores, targets, skip)
            if not torch.isfinite(loss):
                raise TrainingError(f"loss is {loss.item()} at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate_paths(model, val, gallery, ks)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        **{f"val_{k}": round(v, 5) for k, v in metrics.items()},
                        "seconds": round(time.perf_counter() - started, 2)})
        key = selection_key(metrics, config.selection_metric)
        if key > best_key:
            best_key, best_epoch, best_state = key, epoch, snapshot()
        log.info("epoch %2d loss %.4f val R@1 %.4f MRR %.4f (%.1fs)%s", epoch, history[-1]["train_loss"],
                 metrics[SelectionMetric.RECALL_AT_1.value], metrics[SelectionMetric.MRR.value],
                 history[-1]["seconds"], " *" if best_epoch == epoch else "")
    return PathTrainingResult(history, best_epoch, stop_reason, best_state)
