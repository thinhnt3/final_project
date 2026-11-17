"""Train the graph encoder with in-batch InfoNCE against frozen CLIP caption vectors."""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from src.core.exceptions import ConfigError, DataError, TrainingError
from src.embeddings.device import Device
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.models.checkpoint import RunDirectory

log = logging.getLogger(__name__)

# Metrics are compared after rounding so that float noise cannot decide which epoch is best.
_SELECTION_DECIMALS = 6
# Mixed into the seed so that every epoch draws its own, reproducible batches and captions.
_EPOCH_SEED_STRIDE = 100_003
_SECONDS_DECIMALS = 2


class SelectionMetric(StrEnum):
    """Validation metric that decides the best epoch; the other one breaks ties."""

    RECALL_AT_1 = "recall@1"
    MRR = "mrr"


class StopReason(StrEnum):
    """Why training ended."""

    MAX_EPOCHS = "max_epochs"
    NO_IMPROVEMENT = "no_improvement"


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings; see the `training` section of the YAML."""

    seeds: tuple[int, ...]
    device: Device
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    temperature: float
    grad_clip_norm: float
    selection_metric: SelectionMetric

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("training.seeds must be a non-empty list of distinct integers")
        if self.device is Device.AUTO:
            raise ConfigError("training.device must name a device; 'auto' would make runs differ between machines")
        if min(self.batch_size, self.max_epochs, self.patience) < 1:
            raise ConfigError("training.batch_size, max_epochs and patience must be >= 1")
        if min(self.learning_rate, self.temperature, self.grad_clip_norm) <= 0 or self.weight_decay < 0:
            raise ConfigError("training.learning_rate, temperature and grad_clip_norm must be > 0, weight_decay >= 0")


@dataclass(frozen=True)
class RetrievalSet:
    """Images of one split: their sub-graphs and the CLIP vectors of their captions."""

    graphs: Sequence[Data]
    caption_vectors: torch.Tensor  # (images, captions per image, dim), unit length

    def __post_init__(self) -> None:
        if self.caption_vectors.ndim != 3 or self.caption_vectors.shape[0] != len(self.graphs):
            raise DataError(
                f"{len(self.graphs)} graphs but caption vectors have shape {tuple(self.caption_vectors.shape)}"
            )
        if len(self.graphs) < 2:
            raise DataError("a retrieval set needs at least two images")


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one run."""

    history: list[dict[str, Any]]
    best_epoch: int
    stop_reason: StopReason


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so that model initialization is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def info_nce(graph_vectors: torch.Tensor, text_vectors: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric in-batch contrastive loss: row i of each input is a matching pair.

    Every other row of the batch is a negative, in both directions (graph → text and
    text → graph). Inputs are unit vectors.
    """
    if graph_vectors.shape != text_vectors.shape or graph_vectors.ndim != 2:
        raise DataError(f"pair shapes differ: {tuple(graph_vectors.shape)} vs {tuple(text_vectors.shape)}")
    logits = graph_vectors @ text_vectors.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (nn.functional.cross_entropy(logits, targets) + nn.functional.cross_entropy(logits.T, targets)) / 2


def epoch_batches(
    n_images: int, captions_per_image: int, batch_size: int, seed: int, epoch: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batches of (image positions, caption choice) for one epoch.

    Each image appears exactly once per epoch, hence at most once per batch, with one of its
    captions drawn at random. The draw depends only on (seed, epoch), so a resumed run
    repeats exactly what an uninterrupted run would have done.
    """
    generator = torch.Generator().manual_seed(seed * _EPOCH_SEED_STRIDE + epoch)
    order = torch.randperm(n_images, generator=generator)
    captions = torch.randint(captions_per_image, (n_images,), generator=generator)
    starts = list(range(0, n_images, batch_size))
    # A final batch of one image has no negative: its loss is zero. Merge it into the batch before.
    if len(starts) > 1 and n_images - starts[-1] == 1:
        starts.pop()
    ends = [*starts[1:], n_images]
    return [(order[a:b], captions[order[a:b]]) for a, b in zip(starts, ends, strict=True)]


def _pair_loss(
    model: nn.Module, data: RetrievalSet, images: torch.Tensor, captions: torch.Tensor, config: TrainingConfig
) -> torch.Tensor:
    device = config.device.value
    batch = Batch.from_data_list([data.graphs[i] for i in images.tolist()]).to(device)
    texts = data.caption_vectors[images, captions].to(device)
    return info_nce(model(batch), texts, config.temperature)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: RetrievalSet,
    config: TrainingConfig,
    seed: int,
    epoch: int,
    *,
    show_progress: bool = False,
) -> float:
    """One pass over the train images; returns the mean loss per image."""
    model.train()
    batches = epoch_batches(len(data.graphs), data.caption_vectors.shape[1], config.batch_size, seed, epoch)
    total = 0.0
    for images, captions in tqdm(batches, desc=f"epoch {epoch}", leave=False, disable=not show_progress):
        loss = _pair_loss(model, data, images, captions, config)
        if not torch.isfinite(loss):
            raise TrainingError(f"loss is {loss.item()} at epoch {epoch}; stopping before the weights are updated")
        optimizer.zero_grad()
        loss.backward()
        try:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
        except RuntimeError as error:
            raise TrainingError(f"gradient is not finite at epoch {epoch}; the weights were not updated") from error
        optimizer.step()
        total += loss.item() * len(images)
    return total / len(data.graphs)


@torch.no_grad()
def encode_graphs(model: nn.Module, graphs: Sequence[Data], config: TrainingConfig) -> torch.Tensor:
    """Graph vector of every image, in the given order, on the CPU."""
    model.eval()
    device = config.device.value
    parts = [
        model(Batch.from_data_list(list(graphs[i:i + config.batch_size])).to(device)).cpu()
        for i in range(0, len(graphs), config.batch_size)
    ]
    return torch.cat(parts)


@torch.no_grad()
def evaluate(model: nn.Module, data: RetrievalSet, config: TrainingConfig, ks: Sequence[int]) -> dict[str, float]:
    """Loss and graph-only retrieval metrics on a split: every caption against every image."""
    graph_vectors = encode_graphs(model, data.graphs, config)
    n_images, captions_per_image, dim = data.caption_vectors.shape
    scores = (data.caption_vectors.reshape(-1, dim) @ graph_vectors.T).numpy()
    relevant = np.repeat(np.arange(n_images), captions_per_image)
    metrics = retrieval_metrics(ranks_of_relevant(scores, relevant), ks)
    # Loss on fixed pairs (first caption of each image, images in order), so epochs are comparable.
    total = 0.0
    for start in range(0, n_images, config.batch_size):
        rows = slice(start, start + config.batch_size)
        if graph_vectors[rows].shape[0] < 2:
            continue  # a single pair has no negative
        loss = info_nce(graph_vectors[rows], data.caption_vectors[rows, 0], config.temperature)
        total += loss.item() * graph_vectors[rows].shape[0]
    return {"loss": total / n_images, **metrics}


def selection_key(metrics: dict[str, float], selection_metric: SelectionMetric) -> tuple[float, float]:
    """Higher is better: the selection metric first, the other one to break ties."""
    other = SelectionMetric.MRR if selection_metric is SelectionMetric.RECALL_AT_1 else SelectionMetric.RECALL_AT_1
    return (
        round(metrics[selection_metric.value], _SELECTION_DECIMALS),
        round(metrics[other.value], _SELECTION_DECIMALS),
    )


def fit(
    model: nn.Module,
    train: RetrievalSet,
    val: RetrievalSet,
    config: TrainingConfig,
    seed: int,
    run: RunDirectory,
    ks: Sequence[int],
    *,
    resume: bool = False,
    show_progress: bool = False,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Train until validation stops improving; checkpoint after every epoch.

    `last.pt` always holds the last finished epoch, so an interrupted run loses at most the
    epoch in progress. `best.pt` holds the epoch with the best validation selection metric.
    `on_epoch` receives each finished epoch's history row (used to report to a tracker).
    """
    metric_ks = sorted({*ks, 1})
    model.to(config.device.value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    if resume:
        state = run.load_last()
        if state["seed"] != seed:
            raise TrainingError(f"checkpoint was trained with seed {state['seed']}, not {seed}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_epoch = state["history"], state["best_epoch"]
        torch.set_rng_state(state["rng_state"])
        # `best.pt` is written before `last.pt`; a crash in between leaves them out of step.
        if best_epoch > 0 and run.load_best()["epoch"] != best_epoch:
            raise TrainingError(
                f"{run.path}: best.pt and last.pt disagree on the best epoch; start the run again with --overwrite"
            )
        saved_key = state["best_key"]
        best_key = (saved_key[0], saved_key[1]) if saved_key is not None else None
        log.info("resumed after epoch %d (best epoch %d)", len(history), best_epoch)

    stop_reason = StopReason.MAX_EPOCHS
    for epoch in range(len(history) + 1, config.max_epochs + 1):
        # `patience` epochs in a row without a new best: stop before starting another one.
        if epoch - 1 - best_epoch >= config.patience:
            stop_reason = StopReason.NO_IMPROVEMENT
            break
        started = time.monotonic()
        train_loss = train_epoch(model, optimizer, train, config, seed, epoch, show_progress=show_progress)
        metrics = evaluate(model, val, config, metric_ks)
        key = selection_key(metrics, config.selection_metric)
        is_best = best_key is None or key > best_key
        if is_best:
            best_key, best_epoch = key, epoch
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            **{f"val_{name}": value for name, value in metrics.items() if name != "loss"},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.monotonic() - started, _SECONDS_DECIMALS),
            "is_best": is_best,
        })
        model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if is_best:
            run.save_best({"model": model_state, "epoch": epoch, "seed": seed, "metrics": metrics})
        run.save_last({
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "seed": seed,
            "history": history,
            "best_epoch": best_epoch,
            "best_key": list(best_key) if best_key is not None else None,
            "rng_state": torch.get_rng_state(),
        })
        run.save_history(history)
        if on_epoch is not None:
            on_epoch(history[-1])
        log.info(
            "epoch %3d  train loss %.4f  val loss %.4f  val R@1 %.4f  MRR %.4f  lr %.2g  %.1fs%s",
            epoch, train_loss, metrics["loss"], metrics["recall@1"], metrics["mrr"],
            history[-1]["learning_rate"], history[-1]["seconds"], "  *best*" if is_best else "",
        )
    return TrainingResult(history=history, best_epoch=best_epoch, stop_reason=stop_reason)
