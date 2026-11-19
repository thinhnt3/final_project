"""Relation sparsity, keep-only diagnostics and explicitly sourced validity reviews.

Sparsity is measured over relation triplets, not all PyG edges or object nodes.
Semantic validity requires labels; graph membership alone is not a truth label.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from src.core.exceptions import DataError
from src.models.explain import score_graph_variants

LABELS = frozenset({"yes", "no", "uncertain"})


def relation_sparsity(
    total: int,
    selected: Sequence[int],
) -> dict[str, int | float]:
    """Validate a nonempty explanation and measure 1 - selected/available relations."""
    # Positions identify relations even when two triplets have identical text.
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 1
        or not selected
        or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < total for i in selected)
        or len(set(selected)) != len(selected)
    ):
        raise DataError("need a positive relation count and distinct in-range explanation positions")
    size = len(selected)
    return {
        "available_relations": total,
        "explanation_relations": size,
        "explanation_fraction": size / total,
        "sparsity": 1 - size / total,
    }


def retain_relations(
    graph: Mapping[str, Any],
    selected: Sequence[int],
) -> dict[str, Any]:
    """Keep just the explanation triplets; retain all objects and image-object links."""
    relation_sparsity(
        len(graph["triplets"]),
        selected,
    )
    positions = set(selected)
    return {**graph, "triplets": [t for i, t in enumerate(graph["triplets"]) if i in positions]}


def keep_only_scores(
    model: nn.Module,
    graph: Mapping[str, Any],
    image_row: int,
    query: torch.Tensor,
    budget: Mapping[str, Any],
    batch_size: int = 64,
) -> dict[str, Any]:
    """Compare retained selected/random subsets from the existing deletion protocol."""
    selected = budget["selected_indices"]
    controls = budget["random_indices"]
    if not controls or any(len(group) != len(selected) for group in controls):
        raise DataError("keep-only controls must be nonempty and match the explanation budget")

    # Use precisely the same relation subsets for deletion and retention diagnostics.
    variants: list[Mapping[str, Any]] = [
        retain_relations(
            graph,
            positions,
        )
        for positions in [selected, *controls]
    ]
    scores = score_graph_variants(
        model,
        variants,
        image_row,
        query,
        batch_size,
    )
    return {
        "selected_score": float(scores[0]),
        "random_scores": scores[1:].tolist(),
    }


def summarize_quality(
    queries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate queries within each budget; never treat random repeats as queries."""
    fractions = sorted({b["fraction_requested"] for q in queries for b in q.get("budgets", [])})
    summary = []
    for fraction in fractions:
        budgets = [b for q in queries for b in q.get("budgets", []) if b["fraction_requested"] == fraction]
        keys = (
            "available_relations",
            "explanation_relations",
            "explanation_fraction",
            "sparsity",
            "deletion_graph_drop",
            "deletion_top1_lost",
            "keep_graph_drop",
            "keep_random_graph_drop",
            "keep_top1_preserved",
            "keep_random_top1_preserved",
            "keep_rank_increase",
        )
        summary.append(
            {
                "fraction_requested": fraction,
                "evaluated": len(budgets),
                **{key: float(np.mean([b[key] for b in budgets])) for key in keys},
            }
        )
    return summary


def select_review_queries(
    queries: Sequence[Mapping[str, Any]],
    per_stratum: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Seeded equal-size relevant/non-relevant prediction sample, not a population estimate."""
    if isinstance(per_stratum, bool) or per_stratum < 1:
        raise DataError("review_per_stratum must be positive")
    rng = random.Random(seed)
    selected = []
    for correct in (True, False):
        pool = sorted(
            (q for q in queries if q["status"] == "evaluated" and q["predicted_image_is_relevant"] is correct),
            key=lambda q: q["query_id"],
        )
        if len(pool) < per_stratum:
            raise DataError("not enough evaluated queries in a validity review stratum")
        selected.extend(rng.sample(pool, per_stratum))
    return selected


def summarize_validity(
    cases: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete review, retaining unknowns and explicit AI/human provenance."""
    if review.get("reviewer_type") not in {"ai_assisted", "human"} or not review.get("reviewer_id"):
        raise DataError("validity review requires an explicit reviewer type and identity")

    # Require exactly one label per exported relation; reject silent missing/extra cases.
    expected = {(c["case_id"], r["index"]) for c in cases for r in c["relations"]}
    records = review.get("labels")
    if not isinstance(records, list) or not expected or len({c["case_id"] for c in cases}) != len(cases):
        raise DataError("validity review needs nonempty cases and a labels list")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or type(record.get("relation_index")) is not int:
            raise DataError("validity relation index must be an integer")
        key = (record.get("case_id"), record.get("relation_index"))
        if key not in expected or key in seen:
            raise DataError("unknown or duplicate validity relation label")
        if any(record.get(field) not in LABELS for field in ("image_support", "query_relevance")):
            raise DataError("validity labels must be yes, no or uncertain")
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise DataError("each validity label needs an evidence note")
        seen.add(key)
    if seen != expected:
        raise DataError("validity review is incomplete; missing relation labels")

    # A definite failure in either dimension makes joint validity fail, even if the other is unknown.
    def joint(record: Mapping[str, Any]) -> str:
        values = (record["image_support"], record["query_relevance"])
        if "no" in values:
            return "no"
        return "uncertain" if "uncertain" in values else "yes"

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"relations": len(rows)}
        for field in ("image_support", "query_relevance", "joint_validity"):
            values = [joint(r) if field == "joint_validity" else r[field] for r in rows]
            yes, unknown = values.count("yes"), values.count("uncertain")
            n = len(values)
            result[field] = {
                "yes": yes,
                "no": values.count("no"),
                "uncertain": unknown,
                "lower_bound": yes / n if n else None,
                "upper_bound": (yes + unknown) / n if n else None,
                "resolved_fraction": (n - unknown) / n if n else None,
                "yes_among_resolved": yes / (n - unknown) if n > unknown else None,
            }
        return result

    by_case = {c["case_id"]: c for c in cases}
    per_case = [
        {
            "case_id": case_id,
            **aggregate([r for r in records if r["case_id"] == case_id]),
        }
        for case_id in by_case
    ]
    return {
        "reviewer_type": review["reviewer_type"],
        "reviewer_id": review["reviewer_id"],
        "queries": len(cases),
        "micro": aggregate(records),
        "macro_joint_validity_lower_bound": float(np.mean([c["joint_validity"]["lower_bound"] for c in per_case])),
        "per_case": per_case,
        "by_prediction_relevance": {
            str(correct).lower(): aggregate(
                [r for r in records if by_case[r["case_id"]]["predicted_image_is_relevant"] is correct]
            )
            for correct in (True, False)
        },
        "limitation": "Balanced diagnostic sample; reviewer labels are not independent human consensus.",
    }
