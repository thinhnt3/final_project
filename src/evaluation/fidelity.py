"""Joint relation-deletion diagnostics on a frozen graph encoder.

Explanations are ranked by single-deletion score drops, then evaluated by joint
removal. This is a perturbation diagnostic, not independent proof of faithfulness.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from src.core.exceptions import DataError
from src.models.ablations import remove_relations
from src.models.explain import score_graph_variants

# Joint deletion removes at least two relations and keeps at least one.
MIN_RELATIONS = 3


def joint_relation_fidelity(
    model: nn.Module,
    graph: Mapping[str, Any],
    image_row: int,
    query: torch.Tensor,
    fractions: Sequence[float],
    random_repeats: int,
    random_seed: int,
    batch_size: int = 64,
    importance: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compare the top-ranked relations removed jointly against equal-budget random groups.

    Relations are ranked by `importance` (one value per relation, larger = more important)
    when given, e.g. from GNNExplainer; otherwise by their single-deletion score drops.

    At least two relations are removed; at least one remains. Graphs with <3
    relations cannot support this contrast and are reported as skipped. Negative
    drops are kept; ranking never assumes a relation is helpful.
    """
    if not fractions or any(not 0 < f < 1 for f in fractions) or random_repeats < 1:
        raise DataError("need fractions in (0,1) and random_repeats >= 1")

    # Joint deletion needs at least two removed relations and one remaining.
    n = len(graph["triplets"])
    if n < MIN_RELATIONS:
        return {"status": "skipped", "reason": "fewer_than_three_relations", "relations": n}

    # Rank relations by their individual signed score drops.
    single = (
        score_graph_variants(
            model,
            [graph] + [remove_relations(graph, [i]) for i in range(n)],
            image_row,
            query,
            batch_size,
        )
        .numpy()
        .astype(float)
    )
    baseline = float(single[0])
    drops = baseline - single[1:]
    if importance is not None and (len(importance) != n or not np.isfinite(importance).all()):
        raise DataError("importance needs one finite value per relation")
    order = np.argsort(-(drops if importance is None else np.asarray(importance, dtype=float)), kind="stable")
    # Top single-deletion relations at each budget, to report how far another ranking departs from them.
    deletion_order = np.argsort(-drops, kind="stable")
    rng = random.Random(random_seed)
    outputs = []
    for fraction in fractions:
        # Compare selected and random groups with exactly the same budget.
        budget = min(n - 1, max(2, math.ceil(n * fraction)))
        chosen = [int(i) for i in order[:budget]]
        controls = [rng.sample(range(n), budget) for _ in range(random_repeats)]
        subsets = [chosen, *controls]
        scores = (
            score_graph_variants(
                model,
                [remove_relations(graph, indices) for indices in subsets],
                image_row,
                query,
                batch_size,
            )
            .numpy()
            .astype(float)
        )

        # Preserve negative drops: removing a harmful relation can raise the score.
        random_drops = baseline - scores[1:]
        selected_drop = baseline - float(scores[0])
        outputs.append(
            {
                "fraction_requested": float(fraction),
                "removed": budget,
                "fraction_actual": budget / n,
                "selected_indices": chosen,
                "single_drops": [float(drops[i]) for i in chosen],
                "overlap_with_single_deletion": len({*chosen} & {int(i) for i in deletion_order[:budget]}) / budget,
                "random_indices": controls,
                "selected_score": float(scores[0]),
                "random_scores": scores[1:].tolist(),
                "selected_drop": selected_drop,
                "random_drop_mean": float(random_drops.mean()),
                "excess_drop": selected_drop - float(random_drops.mean()),
                "positive_drop": selected_drop > 0,
                "beats_random_mean": selected_drop > float(random_drops.mean()),
            }
        )
    return {"status": "evaluated", "relations": n, "baseline_graph_score": baseline, "budgets": outputs}


def target_rank(
    scores: np.ndarray,
    target: int,
    replacement: float,
) -> int:
    """Conservative rank after changing exactly one target score; all rivals fixed."""
    if scores.ndim != 1 or not 0 <= target < len(scores) or not np.isfinite(replacement):
        raise DataError("need a score vector, valid target and finite replacement")
    return 1 + int(np.count_nonzero(np.delete(scores, target) >= replacement))


def ranking_effects(
    final_scores: np.ndarray,
    target: int,
    clip_score: float,
    alpha: float,
    intervention: Mapping[str, Any],
) -> dict[str, Any]:
    """Frozen-pool rank diagnostics for the target that was predicted before intervention."""
    if not 0 <= alpha <= 1:
        raise DataError("alpha must be in [0,1]")

    # Replace only the target score; leave every competing image untouched.
    before = float(final_scores[target])
    baseline_rank = target_rank(
        final_scores,
        target,
        before,
    )
    selected = alpha * clip_score + (1 - alpha) * float(intervention["selected_score"])
    random_scores = [alpha * clip_score + (1 - alpha) * float(s) for s in intervention["random_scores"]]
    random_ranks = [
        target_rank(
            final_scores,
            target,
            s,
        )
        for s in random_scores
    ]
    selected_rank = target_rank(
        final_scores,
        target,
        selected,
    )
    return {
        "baseline_rank": baseline_rank,
        "selected_rank": selected_rank,
        "selected_rank_increase": selected_rank - baseline_rank,
        "random_rank_increase_mean": float(np.mean(random_ranks)) - baseline_rank,
        "selected_top1_lost": baseline_rank == 1 and selected_rank > 1,
        "random_top1_lost_rate": float(np.mean([baseline_rank == 1 and r > 1 for r in random_ranks])),
        "fused_score_drop": before - selected,
    }
