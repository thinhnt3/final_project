"""Fixed-checkpoint relation fidelity with disjoint, equal-budget random controls."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from src.core.exceptions import ConfigError
from src.models.ablations import remove_relations
from src.models.explain import score_graph_variants


@dataclass(frozen=True)
class FidelityResult:
    """Signed score changes; skipped controls are explicit rather than scored as zero."""

    graph_score: float
    removed_graph_score: float
    control_graph_score: float | None
    selected_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    relation_count: int
    requested_budget: int
    alpha: float
    status: str
    is_candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready values with graph deltas scaled to the unchanged fusion score."""
        weight = (1 - self.alpha) if self.is_candidate else 0.0
        return {
            **asdict(self),
            "final_score_drop": weight * (self.graph_score - self.removed_graph_score),
            "control_final_score_drop": (
                None if self.control_graph_score is None else weight * (self.graph_score - self.control_graph_score)
            ),
            "sparsity": 1 - len(self.selected_indices) / max(1, self.relation_count),
        }


def evaluate_fidelity(
    model: nn.Module,
    image_graph: Mapping[str, Any],
    image_row: int,
    query_vector: torch.Tensor,
    *,
    budget: int,
    seed: int,
    alpha: float,
    is_candidate: bool = True,
) -> FidelityResult:
    """Remove supporting relations jointly and compare with random complementary ones.

    The effective budget is capped at half the relations to permit a disjoint control.
    Positive leave-one-out deltas select support; the joint effect is recomputed rather
    than summed. CLIP, alpha and the image/query stay fixed. No semantic-validity claim.
    """
    if type(budget) is not int or budget < 1 or type(seed) is not int or seed < 0:
        raise ConfigError("budget must be positive and seed must be a nonnegative integer")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ConfigError("alpha must be finite and within [0, 1]")
    count = len(image_graph["triplets"])
    limit = min(budget, count // 2)
    variants = [image_graph] + [remove_relations(image_graph, [i]) for i in range(count)]
    scores = score_graph_variants(model, variants, image_row, query_vector)
    base = scores[0]
    support = sorted((i for i in range(count) if scores[i + 1] < base), key=lambda i: (scores[i + 1], i))
    selected = tuple(support[:limit])
    if not selected:
        status = "insufficient_relations" if limit == 0 else "no_supporting_relations"
        return FidelityResult(base, base, None, (), (), count, budget, alpha, status, is_candidate)
    complement = [i for i in range(count) if i not in selected]
    rng = random.Random(f"{seed}:{image_graph['vg_image_id']}")
    control = tuple(sorted(rng.sample(complement, len(selected))))
    removed, random_removed = score_graph_variants(
        model,
        [remove_relations(image_graph, selected), remove_relations(image_graph, control)],
        image_row,
        query_vector,
    )
    return FidelityResult(
        base, removed, random_removed, selected, control, count, budget, alpha, "controlled", is_candidate
    )
