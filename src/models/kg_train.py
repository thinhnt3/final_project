"""Train the knowledge-graph entity encoder so that a query's CLIP text vector finds its answer entity."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.core.exceptions import DataError, TrainingError
from src.evaluation.entity_retrieval import filtered_ranks
from src.evaluation.metrics import retrieval_metrics
from src.models.train import SelectionMetric, StopReason, TrainingConfig, selection_key

log = logging.getLogger(__name__)

# Mixed into the seed so that every epoch draws its own, reproducible batches and answers.
_EPOCH_SEED_STRIDE = 100_003


@dataclass(frozen=True)
class KgQuerySet:
    """Queries of one split, with answers given as gallery columns.

    `own_triples[i]` lists the rows of the train triple tensor that state query i's own fact.
    They are removed from the graph while the query is trained on, because at test time the
    asked fact is never in the graph. Empty for queries without a triple and for val/test.
    """

    query_vectors: torch.Tensor  # (queries, dim), unit length
    relevant: Sequence[Sequence[int]]
    known: Sequence[Sequence[int]]
    own_triples: Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        n = self.query_vectors.shape[0]
        if self.query_vectors.ndim != 2 or not len(self.relevant) == len(self.known) == len(self.own_triples) == n:
            raise DataError("query vectors, relevant, known and own_triples must describe the same queries")
        if n < 2 or any(not answers for answers in self.relevant):
            raise DataError("a query set needs at least two queries, each with an answer")


@dataclass(frozen=True)
class KgTrainingResult:
    """Outcome of one run; `entity_vectors` come from the best epoch."""

    history: list[dict[str, Any]]
    best_epoch: int
    stop_reason: StopReason
    entity_vectors: np.ndarray
    state_dict: dict[str, torch.Tensor]


def gallery_loss(
    query_vectors: torch.Tensor, gallery_vectors: torch.Tensor, targets: torch.Tensor,
    skip: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Cross-entropy of each query against the whole gallery.

    `targets[i]` is the gallery column of query i's answer. `skip` marks its other correct
    columns, which are taken out of the softmax so they are not pushed away as negatives.
    """
    logits = query_vectors @ gallery_vectors.T / temperature
    return nn.functional.cross_entropy(logits.masked_fill(skip, float("-inf")), targets)


def _batch_inputs(
    data: KgQuerySet, rows: torch.Tensor, n_gallery: int, n_triples: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sampled answer column per query, the skip mask, and which train triples stay in the graph."""
    targets = torch.empty(len(rows), dtype=torch.long)
    skip = torch.zeros(len(rows), n_gallery, dtype=torch.bool)
    keep = torch.ones(n_triples, dtype=torch.bool)
    for position, row in enumerate(rows.tolist()):
        answers = data.relevant[row]
        target = answers[int(torch.randint(len(answers), (1,), generator=generator))]
        targets[position] = target
        others = [column for column in (*answers, *data.known[row]) if column != target]
        skip[position, others] = True
        keep[list(data.own_triples[row])] = False
    return targets, skip, keep


def evaluate(
    model: nn.Module, triples: torch.Tensor, gallery: torch.Tensor, data: KgQuerySet, ks: Sequence[int]
) -> dict[str, float]:
    """Filtered Recall@K and MRR of the graph vectors alone, with the full train graph as edges."""
    model.eval()
    with torch.inference_mode():
        scores = (data.query_vectors @ model(triples)[gallery].T).numpy()
    ranks, _ = filtered_ranks(scores, data.relevant, data.known)
    return retrieval_metrics(ranks, sorted({*ks, 1}))


def fit(
    model: nn.Module,
    triples: torch.Tensor,
    gallery: torch.Tensor,
    train: KgQuerySet,
    val: KgQuerySet,
    config: TrainingConfig,
    seed: int,
    ks: Sequence[int],
) -> KgTrainingResult:
    """Train until validation stops improving; return the best epoch's entity vectors and weights.

    `triples` is the train graph, (triples × 3) head, relation, tail. `gallery` holds the
    entity ids that can be answers, in gallery-column order. Batches depend only on
    (seed, epoch); seed the weights by calling `seed_everything(seed)` before building the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_epoch, best_vectors, best_state = 0, np.empty(0), {}
    stop_reason = StopReason.MAX_EPOCHS
    n_queries = train.query_vectors.shape[0]
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
            targets, skip, keep = _batch_inputs(train, rows, len(gallery), triples.shape[0], generator)
            loss = gallery_loss(
                train.query_vectors[rows], model(triples[keep])[gallery], targets, skip, config.temperature
            )
            if not torch.isfinite(loss):
                raise TrainingError(f"loss is {loss.item()} at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, triples, gallery, val, ks)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        **{f"val_{k}": round(v, 5) for k, v in metrics.items()},
                        "seconds": round(time.perf_counter() - started, 2)})
        key = selection_key(metrics, config.selection_metric)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            with torch.inference_mode():
                best_vectors = model(triples).numpy().astype(np.float32)
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        log.info("epoch %2d loss %.4f val R@1 %.4f MRR %.4f (%.1fs)%s", epoch, history[-1]["train_loss"],
                 metrics[SelectionMetric.RECALL_AT_1.value], metrics[SelectionMetric.MRR.value],
                 history[-1]["seconds"], " *" if best_epoch == epoch else "")
    return KgTrainingResult(history, best_epoch, stop_reason, best_vectors, best_state)
