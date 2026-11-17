"""Paired retrieval diagnostics; uncertainty conditional on the available trained seeds."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.exceptions import DataError


def paired_analysis(
    full_ranks: np.ndarray,
    variant_ranks: np.ndarray,
    image_ids: np.ndarray,
    repeats: int = 2000,
    seed: int = 310,
) -> dict[str, Any]:
    """Cluster-bootstrap true images, keeping captions and model seeds paired.

    The interval measures query/gallery-image sampling uncertainty for these fixed
    checkpoints; it does not establish generalization over new training seeds.
    """
    if (
        full_ranks.ndim != 2
        or full_ranks.shape != variant_ranks.shape
        or not full_ranks.size
        or image_ids.shape != (full_ranks.shape[1],)
        or repeats < 1
        or not np.isfinite(full_ranks).all()
        or not np.isfinite(variant_ranks).all()
        or np.any(full_ranks < 1)
        or np.any(variant_ranks < 1)
    ):
        raise DataError("need paired [seeds,queries] ranks >=1, query image IDs, and repeats >=1")

    # Average paired top-1 differences across the fixed training seeds.
    difference = ((full_ranks == 1).astype(float) - (variant_ranks == 1)).mean(axis=0)

    # Resample images as clusters instead of treating captions as independent.
    unique, clusters = np.unique(image_ids, return_inverse=True)
    sums = np.bincount(clusters, weights=difference)
    counts = np.bincount(clusters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(repeats, len(unique)))
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)

    # Retain wins and losses that would cancel out in the mean.
    per_seed = []
    for full, variant in zip(full_ranks, variant_ranks, strict=True):
        per_seed.append(
            {
                "full_correct_variant_wrong": int(((full == 1) & (variant != 1)).sum()),
                "variant_correct_full_wrong": int(((variant == 1) & (full != 1)).sum()),
                "net_full_top1_count": int((full == 1).sum() - (variant == 1).sum()),
            }
        )
    return {
        "per_seed": per_seed,
        "delta_r1_pp": float(difference.mean() * 100),
        "image_cluster_bootstrap_95ci_pp": (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist(),
        "image_clusters": len(unique),
        "repeats": repeats,
        "bootstrap_seed": seed,
        "scope": "conditional on fixed trained seeds; captions clustered by true image",
    }
