"""Train the graph encoder with in-batch InfoNCE against frozen CLIP caption vectors."""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

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
    # Caption recall@1 plus validation relation accuracy; needs the relation validator.
    RELATION = "recall@1+relation"


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
    # Images per hard-negative group: an image and partners sharing an object pair with it
    # (see `RetrievalSet.partners`) are put in the same batch. 1 switches it off.
    hard_negative_group: int = 1
    # Weight of the predicate-swap term: for a caption that names a triplet of its image, the
    # same graph with that predicate swapped must score below the real one. 0 switches it off.
    predicate_swap_weight: float = 0.0
    # Replacement predicates are drawn from the this many most frequent ones (ids 0..n-1).
    predicate_swap_pool: int = 50
    # Share of the train images used, drawn once with a fixed seed; 1 uses them all.
    train_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("training.seeds must be a non-empty list of distinct integers")
        if self.device is Device.AUTO:
            raise ConfigError("training.device must name a device; 'auto' would make runs differ between machines")
        if min(self.batch_size, self.max_epochs, self.patience, self.hard_negative_group) < 1:
            raise ConfigError("training.batch_size, max_epochs and patience must be >= 1")
        if self.predicate_swap_weight < 0 or self.predicate_swap_pool < 2 or not 0 < self.train_fraction <= 1:
            raise ConfigError("training.predicate_swap_weight >= 0, predicate_swap_pool >= 2, train_fraction in (0, 1]")
        if min(self.learning_rate, self.temperature, self.grad_clip_norm) <= 0 or self.weight_decay < 0:
            raise ConfigError("training.learning_rate, temperature and grad_clip_norm must be > 0, weight_decay >= 0")


@dataclass(frozen=True)
class RetrievalSet:
    """Images of one split: their sub-graphs and the CLIP vectors of their captions."""

    graphs: Sequence[Data]
    caption_vectors: torch.Tensor  # (images, captions per image, dim), unit length
    # For hard-negative batches: per image position, positions of other images to batch it with.
    partners: Sequence[Sequence[int]] | None = None
    # For predicate-swap negatives: per image, per caption, the relation nodes (local indices)
    # whose triplet the caption names.
    relation_mentions: Sequence[Sequence[Sequence[int]]] | None = None

    def __post_init__(self) -> None:
        if self.caption_vectors.ndim != 3 or self.caption_vectors.shape[0] != len(self.graphs):
            raise DataError(
                f"{len(self.graphs)} graphs but caption vectors have shape {tuple(self.caption_vectors.shape)}"
            )
        if len(self.graphs) < 2:
            raise DataError("a retrieval set needs at least two images")
        if self.partners is not None and len(self.partners) != len(self.graphs):
            raise DataError(f"{len(self.graphs)} graphs but {len(self.partners)} partner lists")


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one run."""

    history: list[dict[str, Any]]
    best_epoch: int
    stop_reason: StopReason


@runtime_checkable
class AdversarialEncoder(Protocol):
    """Encoder trained against a generator of synthetic inputs, in alternating steps.

    The encoder is the discriminator: its loss asks real inputs to score above synthetic
    ones, then the generator is updated to close that gap. The generator's parameters belong
    to the encoder module, so checkpoints hold them, but they have their own optimizer.
    """

    adversarial: bool  # False: train as any other encoder
    generator_learning_rate: float

    def generator_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters updated by the generator step only."""
        ...

    def discriminator_loss(self, batch: Batch, texts: torch.Tensor, temperature: float) -> torch.Tensor:
        """Loss of the encoder step: the retrieval loss plus the adversarial term."""
        ...

    def generator_loss(self, batch: Batch, texts: torch.Tensor, temperature: float) -> torch.Tensor:
        """Loss of the generator step."""
        ...


def _adversary(model: nn.Module) -> AdversarialEncoder | None:
    """The model seen as an adversarial encoder, if it is one and the adversary is switched on."""
    return model if isinstance(model, AdversarialEncoder) and model.adversarial else None


def _encoder_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters updated by the encoder step: all of them but an adversary's generator."""
    adversary = _adversary(model)
    generator = {id(p) for p in adversary.generator_parameters()} if adversary is not None else set()
    return [p for p in model.parameters() if id(p) not in generator]


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


def _grouped_order(
    order: torch.Tensor, partners: Sequence[Sequence[int]], group_size: int, generator: torch.Generator
) -> torch.Tensor:
    """`order` rearranged so that each image is followed by up to group_size − 1 unused partners."""
    used = [False] * len(order)
    grouped: list[int] = []
    for image in order.tolist():
        if used[image]:
            continue
        used[image] = True
        free = [p for p in partners[image] if not used[p]]
        picks = [free[i] for i in torch.randperm(len(free), generator=generator)[:group_size - 1].tolist()]
        for p in picks:
            used[p] = True
        grouped += [image, *picks]
    return torch.tensor(grouped)


def epoch_batches(
    n_images: int,
    captions_per_image: int,
    batch_size: int,
    seed: int,
    epoch: int,
    partners: Sequence[Sequence[int]] | None = None,
    group_size: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batches of (image positions, caption choice) for one epoch.

    Each image appears exactly once per epoch, hence at most once per batch, with one of its
    captions drawn at random. With `partners` and group_size > 1, images are laid out in
    groups of an image and its partners, so most groups land in one batch as hard negatives.
    The draw depends only on (seed, epoch), so a resumed run repeats exactly what an
    uninterrupted run would have done.
    """
    generator = torch.Generator().manual_seed(seed * _EPOCH_SEED_STRIDE + epoch)
    order = torch.randperm(n_images, generator=generator)
    captions = torch.randint(captions_per_image, (n_images,), generator=generator)
    if partners is not None and group_size > 1:
        order = _grouped_order(order, partners, group_size, generator)
    starts = list(range(0, n_images, batch_size))
    # A final batch of one image has no negative: its loss is zero. Merge it into the batch before.
    if len(starts) > 1 and n_images - starts[-1] == 1:
        starts.pop()
    ends = [*starts[1:], n_images]
    return [(order[a:b], captions[order[a:b]]) for a, b in zip(starts, ends, strict=True)]


def _pair_inputs(
    data: RetrievalSet, images: torch.Tensor, captions: torch.Tensor, config: TrainingConfig
) -> tuple[Batch, torch.Tensor]:
    """Batched sub-graphs of the given images and the vectors of their chosen captions."""
    device = config.device.value
    batch = Batch.from_data_list([data.graphs[i] for i in images.tolist()]).to(device)
    return batch, data.caption_vectors[images, captions].to(device)


def _swapped_graphs(
    data: RetrievalSet, images: torch.Tensor, captions: torch.Tensor, config: TrainingConfig
) -> tuple[list[int], list[Data]]:
    """Batch rows whose caption names a triplet, and their graphs with that predicate swapped.

    One named relation per row is swapped for another predicate of the pool; the draw uses
    the global generator, which training seeds and checkpoints.
    """
    if data.relation_mentions is None:
        return [], []
    rows, graphs = [], []
    for row, (image, caption) in enumerate(zip(images.tolist(), captions.tolist(), strict=True)):
        named = data.relation_mentions[image][caption]
        if not named:
            continue
        node = named[int(torch.randint(len(named), (1,)))]
        graph = data.graphs[image].clone()
        graph.feature_row = graph.feature_row.clone()
        current = int(graph.feature_row[node])
        replacement = int(torch.randint(config.predicate_swap_pool - 1, (1,)))
        graph.feature_row[node] = replacement + (replacement >= current)  # never the current predicate
        rows.append(row)
        graphs.append(graph)
    return rows, graphs


def _clean_vectors(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Vectors without training-time modality dropout, so two graphs are compared like for like."""
    fused = getattr(model, "fused_vectors", None)
    return fused(batch) if callable(fused) else model(batch)


def _pair_loss(
    model: nn.Module, data: RetrievalSet, images: torch.Tensor, captions: torch.Tensor, config: TrainingConfig
) -> torch.Tensor:
    batch, texts = _pair_inputs(data, images, captions, config)
    adversary = _adversary(model)
    if adversary is not None:
        return adversary.discriminator_loss(batch, texts, config.temperature)
    loss = info_nce(model(batch), texts, config.temperature)
    structure_weight = float(getattr(model, "structure_loss_weight", 0.0))
    if structure_weight > 0:
        # nn.Module's __getattr__ types attributes as Tensor | Module; this one is a method.
        structure_vectors = cast(Callable[[Batch], torch.Tensor], model.structure_vectors)
        loss = loss + structure_weight * info_nce(structure_vectors(batch), texts, config.temperature)
    if config.predicate_swap_weight > 0:
        rows, swapped = _swapped_graphs(data, images, captions, config)
        if rows:
            device = config.device.value
            originals = Batch.from_data_list([data.graphs[i] for i in images[rows].tolist()])
            real = _clean_vectors(model, originals.to(device))
            fake = _clean_vectors(model, Batch.from_data_list(swapped).to(device))
            gap = ((fake - real) * texts[rows]).sum(dim=-1) / config.temperature
            loss = loss + config.predicate_swap_weight * nn.functional.softplus(gap).mean()
    return loss


def _generator_step(
    adversary: AdversarialEncoder,
    optimizer: torch.optim.Optimizer,
    data: RetrievalSet,
    images: torch.Tensor,
    captions: torch.Tensor,
    config: TrainingConfig,
    epoch: int,
) -> float:
    """Update the generator on the batch the encoder was just updated on; returns its loss."""
    loss = adversary.generator_loss(*_pair_inputs(data, images, captions, config), config.temperature)
    if not torch.isfinite(loss):
        raise TrainingError(f"generator loss is {loss.item()} at epoch {epoch}")
    optimizer.zero_grad()
    loss.backward()
    try:
        nn.utils.clip_grad_norm_(list(adversary.generator_parameters()), config.grad_clip_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        raise TrainingError(f"generator gradient is not finite at epoch {epoch}") from error
    optimizer.step()
    return float(loss.item())


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: RetrievalSet,
    config: TrainingConfig,
    seed: int,
    epoch: int,
    *,
    show_progress: bool = False,
    generator_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass over the train images; returns the mean loss per image as history fields.

    With `generator_optimizer`, the model is an adversarial encoder: each batch updates the
    encoder, then the generator, and the generator's mean loss is returned too.
    """
    model.train()
    batches = epoch_batches(
        len(data.graphs), data.caption_vectors.shape[1], config.batch_size, seed, epoch,
        data.partners, config.hard_negative_group,
    )
    encoder_parameters = _encoder_parameters(model)
    adversary = _adversary(model) if generator_optimizer is not None else None
    total, generator_total = 0.0, 0.0
    for images, captions in tqdm(batches, desc=f"epoch {epoch}", leave=False, disable=not show_progress):
        loss = _pair_loss(model, data, images, captions, config)
        if not torch.isfinite(loss):
            raise TrainingError(f"loss is {loss.item()} at epoch {epoch}; stopping before the weights are updated")
        optimizer.zero_grad()
        loss.backward()
        try:
            nn.utils.clip_grad_norm_(encoder_parameters, config.grad_clip_norm, error_if_nonfinite=True)
        except RuntimeError as error:
            raise TrainingError(f"gradient is not finite at epoch {epoch}; the weights were not updated") from error
        optimizer.step()
        total += loss.item() * len(images)
        if adversary is not None and generator_optimizer is not None:
            generator_loss = _generator_step(adversary, generator_optimizer, data, images, captions, config, epoch)
            generator_total += generator_loss * len(images)
    losses = {"train_loss": total / len(data.graphs)}
    if adversary is not None:
        losses["train_generator_loss"] = generator_total / len(data.graphs)
    return losses


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
    if selection_metric is SelectionMetric.RELATION:
        if "relation_acc" not in metrics:
            raise TrainingError("selection on recall@1+relation needs the relation validator")
        combined = metrics[SelectionMetric.RECALL_AT_1.value] + metrics["relation_acc"]
        return round(combined, _SELECTION_DECIMALS), round(metrics[SelectionMetric.MRR.value], _SELECTION_DECIMALS)
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
    extra_validation: Callable[[nn.Module], dict[str, float]] | None = None,
) -> TrainingResult:
    """Train until validation stops improving; checkpoint after every epoch.

    `last.pt` always holds the last finished epoch, so an interrupted run loses at most the
    epoch in progress. `best.pt` holds the epoch with the best validation selection metric.
    `on_epoch` receives each finished epoch's history row (used to report to a tracker).
    `extra_validation` adds metrics to each epoch's validation (e.g. the relation validator).
    """
    metric_ks = sorted({*ks, 1})
    model.to(config.device.value)
    optimizer = torch.optim.AdamW(
        _encoder_parameters(model), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    adversary = _adversary(model)
    generator_optimizer = (
        torch.optim.AdamW(adversary.generator_parameters(), lr=adversary.generator_learning_rate, weight_decay=0.0)
        if adversary is not None
        else None
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    if resume:
        state = run.load_last()
        if state["seed"] != seed:
            raise TrainingError(f"checkpoint was trained with seed {state['seed']}, not {seed}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if generator_optimizer is not None:
            generator_optimizer.load_state_dict(state["generator_optimizer"])
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
        losses = train_epoch(
            model, optimizer, train, config, seed, epoch,
            show_progress=show_progress, generator_optimizer=generator_optimizer,
        )
        train_loss = losses["train_loss"]
        metrics = evaluate(model, val, config, metric_ks)
        if extra_validation is not None:
            metrics.update(extra_validation(model))
        key = selection_key(metrics, config.selection_metric)
        is_best = best_key is None or key > best_key
        if is_best:
            best_key, best_epoch = key, epoch
        history.append({
            "epoch": epoch,
            **losses,
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
            **({"generator_optimizer": generator_optimizer.state_dict()} if generator_optimizer is not None else {}),
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
