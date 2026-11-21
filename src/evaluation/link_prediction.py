"""Filtered link prediction: rank the true head and the true tail of each triple among all entities."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from src.evaluation.entity_retrieval import filtered_ranks
from src.evaluation.metrics import retrieval_metrics

HITS_KS = (1, 3, 10)

# (anchor ids, relation ids, anchor is the head) → (rows × entities) scores of every entity as the other end.
ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def known_completions(triples: Sequence[Sequence[int]]) -> dict[tuple[int, int, bool], set[int]]:
    """For every (entity, relation, entity is the head): the entities that truly complete it."""
    known: dict[tuple[int, int, bool], set[int]] = defaultdict(set)
    for head, relation, tail in triples:
        known[(head, relation, True)].add(tail)
        known[(tail, relation, False)].add(head)
    return known


def link_prediction_metrics(score: ScoreFn, triples: Sequence[Sequence[int]],
                            known: dict[tuple[int, int, bool], set[int]], batch_size: int) -> dict[str, float]:
    """MRR and Hits@K over tail and head prediction; other true completions are skipped."""
    cases = [(h, r, True, t) for h, r, t in triples] + [(t, r, False, h) for h, r, t in triples]
    ranks = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        anchors, relations, is_head, answers = (np.array(column) for column in zip(*batch, strict=True))
        scores = score(anchors, relations, is_head.astype(bool))
        others = [sorted(known[(a, r, bool(d))] - {answer}) for a, r, d, answer in batch]
        ranks.append(filtered_ranks(scores, [[int(answer)] for answer in answers], others)[0])
    metrics = retrieval_metrics(np.concatenate(ranks), HITS_KS)
    return {"mrr": metrics["mrr"], **{f"hits@{k}": metrics[f"recall@{k}"] for k in HITS_KS}}
