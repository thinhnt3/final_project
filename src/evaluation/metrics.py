from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant))


def reciprocal_rank(ranked_ids: list[int], relevant: set[int]) -> float:
    for i, image_id in enumerate(ranked_ids, start=1):
        if image_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(rows: Iterable[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    result = {f"recall@{k}": sum(recall_at_k(r["ranked_ids"], set(r["relevant_ids"]), k) for r in rows) / len(rows) for k in ks}
    result["mrr"] = sum(reciprocal_rank(r["ranked_ids"], set(r["relevant_ids"])) for r in rows) / len(rows)
    return result
