"""Rerank the top-k CLIP candidates with a graph score, and pick alpha and k on validation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.exceptions import ConfigError, DataError
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData, score_method
from src.evaluation.metrics import DEFAULT_KS, ranks_of_relevant, retrieval_metrics

# Both scores must be cosine-like, in [-1, 1] up to rounding.
_SCORE_BOUND = 1.0 + 1e-5
# Subtracted from the CLIP score of images outside the top-k. With scores in [-1, 1], any
# offset above 2 keeps every non-candidate below every candidate while preserving their order.
_NON_CANDIDATE_OFFSET = 3.0
# Metrics are compared after rounding so that float noise cannot decide the selection.
_SELECTION_DECIMALS = 6


def _checked_scores(clip_scores: np.ndarray, graph_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate the two score matrices and return them as float64.

    float32 cannot hold `score − offset` exactly: distinct CLIP scores would collapse into
    ties and change ranks. A float32 value minus the offset is exact in float64.
    """
    if clip_scores.shape != graph_scores.shape or clip_scores.ndim != 2:
        raise DataError(f"score shapes differ: {clip_scores.shape} vs {graph_scores.shape}")
    for name, scores in (("clip", clip_scores), ("graph", graph_scores)):
        if not np.isfinite(scores).all() or np.abs(scores).max() > _SCORE_BOUND:
            raise DataError(f"{name} scores must be finite and within [-1, 1]")
    return clip_scores.astype(np.float64, copy=False), graph_scores.astype(np.float64, copy=False)


def _candidate_mask(clip_scores: np.ndarray, top_k: int) -> np.ndarray:
    """True for each query's top-k CLIP images; ties at the cut all count as candidates."""
    n_items = clip_scores.shape[1]
    if not 1 <= top_k <= n_items:
        raise ConfigError(f"top_k must be in [1, {n_items}], got {top_k}")
    cutoff = np.partition(clip_scores, n_items - top_k, axis=1)[:, n_items - top_k]
    return clip_scores >= cutoff[:, None]


def _fuse(clip_scores: np.ndarray, graph_scores: np.ndarray, is_candidate: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError(f"alpha must be in [0, 1], got {alpha}")
    fused = alpha * clip_scores + (1.0 - alpha) * graph_scores
    return np.where(is_candidate, fused, clip_scores - _NON_CANDIDATE_OFFSET)


def rerank_scores(clip_scores: np.ndarray, graph_scores: np.ndarray, alpha: float, top_k: int) -> np.ndarray:
    """Final score matrix: `alpha·clip + (1−alpha)·graph` for each query's top-k CLIP images.

    Images outside the top-k keep their CLIP order below all candidates. Both inputs must lie
    in [-1, 1]. With `alpha = 1` every item keeps exactly its CLIP rank.
    """
    clip, graph = _checked_scores(clip_scores, graph_scores)
    return _fuse(clip, graph, _candidate_mask(clip, top_k), alpha)


@dataclass(frozen=True)
class SweepPoint:
    """Metrics of one (alpha, top_k) setting."""

    alpha: float
    top_k: int
    metrics: dict[str, float]

    def selection_key(self) -> tuple[float, float, float, int]:
        """Higher is better: Recall@1, then MRR, then more CLIP weight, then fewer candidates."""
        return (
            round(self.metrics["recall@1"], _SELECTION_DECIMALS),
            round(self.metrics["mrr"], _SELECTION_DECIMALS),
            self.alpha,
            -self.top_k,
        )


def sweep_alpha_and_k(
    clip_scores: np.ndarray,
    graph_scores: np.ndarray,
    data: SplitData,
    alphas: Sequence[float],
    top_ks: Sequence[int],
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[SweepPoint, list[SweepPoint]]:
    """Score every (alpha, top_k) pair on the validation split; return the best and all points.

    Best = highest Recall@1, then highest MRR. On a tie the setting closest to plain CLIP
    wins (highest alpha, then smallest top_k), so a graph score that adds nothing is never
    reported as an improvement. Refuses any split other than validation.
    """
    if data.split is not Split.VAL:
        raise DataError(f"alpha and top_k are selected on validation only, got split '{data.split.value}'")
    if not alphas or not top_ks:
        raise ConfigError("alpha grid and top_k grid must not be empty")
    clip, graph = _checked_scores(clip_scores, graph_scores)
    metric_ks = sorted({*ks, 1})  # Recall@1 is always needed for selection
    points = []
    for top_k in top_ks:
        is_candidate = _candidate_mask(clip, top_k)
        for alpha in alphas:
            ranks = ranks_of_relevant(_fuse(clip, graph, is_candidate, alpha), data.relevant_columns)
            points.append(SweepPoint(alpha=alpha, top_k=top_k, metrics=retrieval_metrics(ranks, metric_ks)))
    return max(points, key=SweepPoint.selection_key), points


@dataclass(frozen=True)
class RerankEvaluation:
    """A graph score evaluated on its own and fused with CLIP at the validation-selected setting."""

    selected: SweepPoint
    sweep: list[SweepPoint]
    graph_only: dict[Split, dict[str, float]]
    reranked: dict[Split, dict[str, float]]


def evaluate_reranking(
    clip_scores: Mapping[Split, np.ndarray],
    graph_scores: Mapping[Split, np.ndarray],
    data: Mapping[Split, SplitData],
    alphas: Sequence[float],
    top_ks: Sequence[int],
    ks: Sequence[int] = DEFAULT_KS,
) -> RerankEvaluation:
    """Pick alpha and top_k on validation, then score every given split with that one setting."""
    if Split.VAL not in data:
        raise DataError("reranking needs the validation split to select alpha and top_k")
    missing = [split.value for split in data if split not in clip_scores or split not in graph_scores]
    if missing:
        raise DataError(f"no CLIP or graph scores for splits {missing}")
    smallest_gallery = min(len(split_data.image_ids) for split_data in data.values())
    too_large = [top_k for top_k in top_ks if top_k > smallest_gallery]
    if too_large:
        raise ConfigError(f"top_k values {too_large} exceed the smallest gallery ({smallest_gallery} images)")
    selected, sweep = sweep_alpha_and_k(
        clip_scores[Split.VAL], graph_scores[Split.VAL], data[Split.VAL], alphas, top_ks, ks
    )
    graph_only, reranked = {}, {}
    for split, split_data in data.items():
        graph_only[split] = score_method(graph_scores[split], split_data, ks)
        fused = rerank_scores(clip_scores[split], graph_scores[split], selected.alpha, selected.top_k)
        reranked[split] = score_method(fused, split_data, ks)
    return RerankEvaluation(selected=selected, sweep=sweep, graph_only=graph_only, reranked=reranked)
