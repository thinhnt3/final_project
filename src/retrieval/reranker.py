"""Graph-aware text-to-image reranking with traceable explanations."""
from __future__ import annotations

import re
from typing import Any

import numpy as np


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def graph_score(query: str, graph: dict[str, Any]) -> tuple[float, list[dict[str, str]]]:
    """Score lexical query/graph overlap and return the contributing relations.

    This is the deterministic smoke-test scorer. It is deliberately isolated so a
    learned GCN/GAT/GraphSAGE scorer can replace it without changing the reranker API.
    """
    q = tokenize(query)
    traces = []
    for rel in graph["relationships"]:
        relation_tokens = tokenize(f"{rel['subject']} {rel['predicate']} {rel['object']}")
        overlap = q & relation_tokens
        if overlap:
            traces.append({**rel, "matched_tokens": ",".join(sorted(overlap))})
    object_overlap = q & set(graph["objects"].values())
    score = min(1.0, (len(object_overlap) + 2 * len(traces)) / max(1, len(q)))
    return float(score), traces


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def rerank(query: str, candidate_ids: list[int], clip_scores: np.ndarray,
           graphs: dict[int, dict[str, Any]], alpha: float = 0.7) -> list[dict[str, Any]]:
    graph_scores, explanations = [], []
    for image_id in candidate_ids:
        score, trace = graph_score(query, graphs[image_id])
        graph_scores.append(score)
        explanations.append(trace)
    final = alpha * minmax(np.asarray(clip_scores, dtype=float)) + (1 - alpha) * np.asarray(graph_scores)
    order = np.argsort(-final)
    return [{"image_id": candidate_ids[i], "clip_score": float(clip_scores[i]),
             "graph_score": float(graph_scores[i]), "final_score": float(final[i]),
             "explanation": explanations[i]} for i in order]
