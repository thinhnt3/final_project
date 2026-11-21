"""Query-adaptive fusion of CLIP and graph scores. No file I/O except the config loader.

The official reranker gives every query the same (alpha, top_k). Here the queries are split
into buckets by a label-free feature, and each bucket gets its own (alpha, top_k), selected
on validation with the official sweep and tie-break rules.

A query's fused scores depend only on its own row, so choosing the best setting of every
bucket separately is the same as maximising the overall validation score over all per-bucket
settings. With one bucket, the selection is exactly the official one.

Bucket features:
- `clip_margin`: CLIP top-1 score minus CLIP top-2 score of the query. A small margin means
  CLIP is unsure. The hypothesis under test: the graph score helps most there, so those
  queries should get a lower alpha.
- `random`: a per-query pseudo-random number. It is a control: the same bucket sizes and the
  same number of selected parameters, but no information about the query. A gain that the
  random control also shows comes from overfitting validation, not from the feature.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config_parsing import read_mapping, require_field
from src.core.exceptions import ConfigError, DataError
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData
from src.evaluation.metrics import DEFAULT_KS
from src.retrieval.rerank import SweepPoint, rerank_scores, sweep_alpha_and_k


class BucketFeature(StrEnum):
    """Label-free per-query feature that decides which bucket a query falls in."""

    CLIP_MARGIN = "clip_margin"
    RANDOM = "random"


def clip_margin(clip_scores: np.ndarray) -> np.ndarray:
    """CLIP top-1 score minus top-2 score of every query (rows are queries)."""
    if clip_scores.ndim != 2 or clip_scores.shape[1] < 2:
        raise DataError(f"need a (queries × items) matrix with at least two items, got {clip_scores.shape}")
    if not np.isfinite(clip_scores).all():
        raise DataError("CLIP scores contain NaN or infinite values")
    top_two = np.partition(clip_scores.astype(np.float64, copy=False), -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def random_feature(query_ids: Sequence[str], seed: int) -> np.ndarray:
    """A number in [0, 1) per query, drawn from a private RNG seeded with (seed, query id).

    The value of a query does not depend on the order of the queries or on which split they
    come from, so validation and test draw from the same distribution.
    """
    return np.array([random.Random(f"{seed}:{query_id}").random() for query_id in query_ids], dtype=np.float64)


def query_feature(feature: BucketFeature, clip_scores: np.ndarray, query_ids: Sequence[str], seed: int) -> np.ndarray:
    """The bucket feature of every query; `seed` only affects the random control."""
    if clip_scores.ndim != 2 or clip_scores.shape[0] != len(query_ids):
        raise DataError(f"{len(query_ids)} query ids but CLIP scores have shape {clip_scores.shape}")
    if feature is BucketFeature.CLIP_MARGIN:
        return clip_margin(clip_scores)
    if feature is BucketFeature.RANDOM:
        return random_feature(query_ids, seed)
    raise ConfigError(f"bucket feature '{feature}' has no implementation")


def bucket_edges(values: np.ndarray, n_buckets: int) -> tuple[float, ...]:
    """Interior quantile edges of the (validation) feature values.

    Equal quantiles collapse into one edge, so heavily tied values can yield fewer buckets
    than asked for; the caller sees that from the number of edges.
    """
    if n_buckets < 1:
        raise ConfigError(f"n_buckets must be >= 1, got {n_buckets}")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise DataError("bucket edges need a non-empty, finite, one-dimensional feature")
    if n_buckets == 1:
        return ()
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
    return tuple(float(edge) for edge in np.unique(quantiles))


def assign_buckets(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bucket of each value: bucket i holds edges[i-1] < value <= edges[i]."""
    if not edges:
        return np.zeros(len(values), dtype=np.int64)
    return np.digitize(values, np.asarray(edges, dtype=np.float64), right=True).astype(np.int64)


@dataclass(frozen=True)
class AdaptiveFusion:
    """A fitted bucketing: the feature, its edges cut on validation, and the selected setting of each bucket."""

    feature: BucketFeature
    feature_seed: int
    edges: tuple[float, ...]
    settings: tuple[SweepPoint, ...]

    def __post_init__(self) -> None:
        if len(self.settings) != len(self.edges) + 1:
            raise DataError(f"{len(self.edges)} edges need {len(self.edges) + 1} settings, got {len(self.settings)}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "feature": self.feature.value,
            "feature_seed": self.feature_seed,
            "edges": list(self.edges),
            "buckets": [{"alpha": setting.alpha, "top_k": setting.top_k} for setting in self.settings],
        }


def _subset(data: SplitData, rows: np.ndarray) -> SplitData:
    """The same gallery with only the selected queries."""
    query_ids = tuple(query_id for query_id, keep in zip(data.query_ids, rows, strict=True) if keep)
    return replace(
        data, query_ids=query_ids, relevant_columns=data.relevant_columns[rows], query_vectors=data.query_vectors[rows]
    )


def fit_adaptive_fusion(
    clip_scores: np.ndarray,
    graph_scores: np.ndarray,
    data: SplitData,
    feature: BucketFeature,
    n_buckets: int,
    alphas: Sequence[float],
    top_ks: Sequence[int],
    *,
    min_bucket_queries: int,
    feature_seed: int,
    ks: Sequence[int] = DEFAULT_KS,
) -> AdaptiveFusion:
    """Cut buckets on validation and select (alpha, top_k) per bucket with the official sweep.

    Refuses any split other than validation, and buckets with fewer than
    `min_bucket_queries` validation queries, which are too small to select a setting on.
    """
    if data.split is not Split.VAL:
        raise DataError(f"adaptive fusion is fitted on validation only, got split '{data.split.value}'")
    expected = (len(data.query_ids), len(data.image_ids))
    if clip_scores.shape != expected or graph_scores.shape != expected:
        raise DataError(f"score shapes {clip_scores.shape} and {graph_scores.shape} do not match {expected}")
    if min_bucket_queries < 1:
        raise ConfigError(f"min_bucket_queries must be >= 1, got {min_bucket_queries}")

    values = query_feature(feature, clip_scores, data.query_ids, feature_seed)
    edges = bucket_edges(values, n_buckets)
    buckets = assign_buckets(values, edges)
    settings = []
    for bucket in range(len(edges) + 1):
        rows = buckets == bucket
        count = int(rows.sum())
        if count < min_bucket_queries:
            raise ConfigError(
                f"bucket {bucket} of {feature.value} with {n_buckets} buckets has {count} validation queries, "
                f"fewer than min_bucket_queries={min_bucket_queries}"
            )
        best, _ = sweep_alpha_and_k(clip_scores[rows], graph_scores[rows], _subset(data, rows), alphas, top_ks, ks)
        settings.append(best)
    return AdaptiveFusion(feature=feature, feature_seed=feature_seed, edges=edges, settings=tuple(settings))


def adaptive_scores(
    fusion: AdaptiveFusion,
    clip_scores: np.ndarray,
    graph_scores: np.ndarray,
    query_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fused score matrix (float64) and the bucket of every query.

    Uses the edges and settings fitted on validation as they are; nothing is fitted on the
    split being scored. Each query is fused with the official rule and its bucket's setting.
    """
    if clip_scores.shape != graph_scores.shape or clip_scores.ndim != 2:
        raise DataError(f"score shapes differ: {clip_scores.shape} vs {graph_scores.shape}")
    values = query_feature(fusion.feature, clip_scores, query_ids, fusion.feature_seed)
    buckets = assign_buckets(values, fusion.edges)
    fused = np.empty(clip_scores.shape, dtype=np.float64)
    for bucket, setting in enumerate(fusion.settings):
        rows = buckets == bucket
        if rows.any():
            fused[rows] = rerank_scores(clip_scores[rows], graph_scores[rows], setting.alpha, setting.top_k)
    return fused, buckets


@dataclass(frozen=True)
class AdaptiveFusionConfig:
    """Protocol of `scripts/evaluate_adaptive_fusion.py`; see `configs/evaluation/adaptive_fusion.yaml`."""

    retrieval_config: str
    features: tuple[BucketFeature, ...]
    n_buckets: tuple[int, ...]
    min_bucket_queries: int
    feature_seed: int
    bootstrap_repeats: int
    bootstrap_seed: int
    output: str

    def __post_init__(self) -> None:
        if not self.features or len(set(self.features)) != len(self.features):
            raise ConfigError("features must be a non-empty list without duplicates")
        if not self.n_buckets or len(set(self.n_buckets)) != len(self.n_buckets) or min(self.n_buckets) < 2:
            raise ConfigError("n_buckets must be distinct integers >= 2; the fixed-alpha baseline always runs")
        if self.min_bucket_queries < 1 or self.bootstrap_repeats < 1:
            raise ConfigError("min_bucket_queries and bootstrap_repeats must be >= 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AdaptiveFusionConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        return cls.from_dict(read_mapping(path))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AdaptiveFusionConfig:
        """Build from an already-parsed mapping (used by tests)."""
        mapping = dict(raw)
        features = mapping.get("features")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise ConfigError("features must be a list of names")
        known = {f.value for f in BucketFeature}
        unknown = [f for f in features if f not in known]
        if unknown:
            raise ConfigError(f"unknown features {unknown}; choose from {sorted(known)}")
        n_buckets = mapping.get("n_buckets")
        if not isinstance(n_buckets, list) or not all(
            isinstance(b, int) and not isinstance(b, bool) for b in n_buckets
        ):
            raise ConfigError("n_buckets must be a list of integers")
        return cls(
            retrieval_config=require_field(mapping, "retrieval_config", str),
            features=tuple(BucketFeature(f) for f in features),
            n_buckets=tuple(n_buckets),
            min_bucket_queries=require_field(mapping, "min_bucket_queries", int),
            feature_seed=require_field(mapping, "feature_seed", int),
            bootstrap_repeats=require_field(mapping, "bootstrap_repeats", int),
            bootstrap_seed=require_field(mapping, "bootstrap_seed", int),
            output=require_field(mapping, "output", str),
        )
