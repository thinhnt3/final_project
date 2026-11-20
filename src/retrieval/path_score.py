"""Graph score without training: how strongly train-graph paths connect a query's anchor to each entity."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from src.core.exceptions import DataError


def normalized_adjacency(edges: Sequence[tuple[int, int]], n_entities: int) -> sparse.csr_matrix:
    """Undirected adjacency D^-1/2 A D^-1/2 of the given (head, tail) pairs; parallel edges count once.

    The normalization divides each edge by the square roots of its two degrees, so a path
    through a hub (a node linked to thousands of others) carries little weight.
    """
    if not edges:
        raise DataError("no edges to build the graph from")
    pairs = np.asarray(edges, dtype=np.int64)
    if pairs.min() < 0 or pairs.max() >= n_entities:
        raise DataError("an edge names an entity outside the graph")
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_entities, n_entities))
    adjacency.data[:] = 1.0  # duplicates were summed by the constructor
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    return sparse.diags(scale) @ adjacency @ sparse.diags(scale)


def path_scores(
    adjacency: sparse.csr_matrix, anchors: Sequence[int], max_hops: int, decay: float
) -> np.ndarray:
    """(anchors × entities) score: sum over n = 1..max_hops of decay^n · (weight of n-edge walks).

    Each row is divided by its largest value, so scores lie in [0, 1] and the best-connected
    entity gets 1. The anchor's own column is 0: a query never asks for the entity it names.
    A row is all zeros when the anchor has no edge.
    """
    if max_hops < 1 or not 0.0 < decay <= 1.0:
        raise DataError(f"need max_hops >= 1 and decay in (0, 1], got {max_hops}, {decay}")
    anchor_rows = np.asarray(anchors, dtype=np.int64)
    walk = adjacency[anchor_rows].toarray()
    total = decay * walk
    for hop in range(2, max_hops + 1):
        walk = np.asarray(walk @ adjacency)
        total += decay ** hop * walk
    total[np.arange(len(anchor_rows)), anchor_rows] = 0.0
    peak = total.max(axis=1, keepdims=True)
    return np.divide(total, peak, out=np.zeros_like(total), where=peak > 0).astype(np.float32)


def shuffled_tails(edges: Sequence[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Control graph: the same heads and the same tails, paired at random.

    Every node keeps its number of head and tail slots, so degrees barely change while the
    information about who is linked to whom is destroyed.
    """
    tails = np.asarray([tail for _, tail in edges])
    np.random.default_rng(seed).shuffle(tails)
    return [(head, int(tail)) for (head, _), tail in zip(edges, tails, strict=True)]
