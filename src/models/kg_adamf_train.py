"""Training loop of AdaMF-MAT on a knowledge graph: negative sampling, then the generator step."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.core.exceptions import ConfigError, TrainingError
from src.embeddings.device import Device
from src.evaluation.link_prediction import ScoreFn, known_completions, link_prediction_metrics
from src.models.kg_adamf_mat import KgAdamfMat
from src.models.train import StopReason

log = logging.getLogger(__name__)

_EPOCH_SEED_STRIDE = 100_003


@dataclass(frozen=True)
class KgAdamfTrainingConfig:
    """Optimization settings; see the `adamf_mat` section of the YAML."""

    device: Device
    batch_size: int
    learning_rate: float
    max_epochs: int
    eval_every: int       # epochs between two validation runs
    patience: int         # validation runs without a better MRR before stopping
    eval_batch_size: int

    def __post_init__(self) -> None:
        if self.device is Device.AUTO:
            raise ConfigError("adamf_mat.device must name a device; 'auto' would make runs differ between machines")
        if min(self.batch_size, self.max_epochs, self.eval_every, self.patience, self.eval_batch_size) < 1:
            raise ConfigError("adamf_mat batch sizes, max_epochs, eval_every and patience must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("adamf_mat.learning_rate must be > 0")


@dataclass(frozen=True)
class KgAdamfTrainingResult:
    """Outcome of one run; `state_dict` comes from the best validation run."""

    history: list[dict[str, Any]]
    best_epoch: int
    stop_reason: StopReason
    state_dict: dict[str, torch.Tensor]


def score_fn(model: KgAdamfMat) -> ScoreFn:
    """The model as a link-prediction scorer over every entity."""
    device = model.entities.weight.device
    everyone = torch.arange(model.n_entities, device=device)

    def score(anchors: np.ndarray, relations: np.ndarray, anchor_is_head: np.ndarray) -> np.ndarray:
        model.eval()
        scores = model.candidate_scores(torch.as_tensor(anchors, device=device),
                                        torch.as_tensor(relations, device=device),
                                        torch.as_tensor(anchor_is_head, device=device), everyone)
        return scores.cpu().numpy()

    return score


def fit_kg_adamf(
    model: KgAdamfMat,
    train: Sequence[Sequence[int]],
    val: Sequence[Sequence[int]],
    known: Sequence[Sequence[int]],
    config: KgAdamfTrainingConfig,
    seed: int,
) -> KgAdamfTrainingResult:
    """Train until validation MRR stops improving; return the weights of the best validation run.

    `known` holds every true triple (train, validation, test); it filters the validation ranks.
    """
    device = torch.device(config.device.value)
    model.to(device)
    triples = torch.tensor(train, dtype=torch.long, device=device)
    completions = known_completions(known)
    optimizer = torch.optim.Adam(model.encoder_parameters(), lr=config.learning_rate)
    generator_optimizer = (torch.optim.Adam(model.generator_parameters(), lr=model.config.generator_learning_rate)
                           if model.config.mat else None)
    history: list[dict[str, Any]] = []
    best_mrr, best_epoch, bad_runs = -1.0, 0, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    stop_reason = StopReason.MAX_EPOCHS
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        generator = torch.Generator().manual_seed(seed * _EPOCH_SEED_STRIDE + epoch)
        order = torch.randperm(len(triples), generator=generator).to(device)
        model.train()
        losses, generator_losses = [], []
        for step, start in enumerate(range(0, len(triples), config.batch_size)):
            batch = triples[order[start:start + config.batch_size]]
            negatives = torch.randint(model.n_entities, (len(batch), model.config.negatives),
                                      generator=generator).to(device)
            loss = model.encoder_loss(batch, negatives, corrupt_head=step % 2 == 0)
            if not torch.isfinite(loss):
                raise TrainingError(f"loss is {loss.item()} at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if generator_optimizer is not None:
                generator_loss = model.generator_loss(batch)
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
                generator_losses.append(generator_loss.item())
        if epoch % config.eval_every and epoch != config.max_epochs:
            continue
        metrics = link_prediction_metrics(score_fn(model), val, completions, config.eval_batch_size)
        entry = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                 **{f"val_{k}": round(v, 5) for k, v in metrics.items()},
                 "seconds": round(time.perf_counter() - started, 1)}
        if generator_losses:
            entry["generator_loss"] = round(float(np.mean(generator_losses)), 5)
        history.append(entry)
        improved = metrics["mrr"] > best_mrr
        if improved:
            best_mrr, best_epoch, bad_runs = metrics["mrr"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad_runs += 1
        log.info("epoch %4d loss %.4f val MRR %.4f H@1 %.4f H@10 %.4f (%.0fs)%s", epoch, entry["train_loss"],
                 metrics["mrr"], metrics["hits@1"], metrics["hits@10"], entry["seconds"], " *" if improved else "")
        if bad_runs >= config.patience:
            stop_reason = StopReason.NO_IMPROVEMENT
            break
    return KgAdamfTrainingResult(history, best_epoch, stop_reason, best_state)
