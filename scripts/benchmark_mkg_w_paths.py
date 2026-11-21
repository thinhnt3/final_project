#!/usr/bin/env python3
"""Rerank CLIP on MKG-W with the training-free path score, and with a shuffled-edge control.

Run from the repository root after `embed_mkg_w`:
    python -m scripts.benchmark_mkg_w_paths

Final score = alpha · CLIP + (1 − alpha) · path score over the top-k CLIP entities. Alpha,
top-k and the path decay are picked on validation, separately for the real graph and for
the control graph. The anchor of a relation query is read from the query record
(`anchor_id`), i.e. entity linking is assumed solved. Queries without an anchor get a zero
path score. Results go to <runs_dir>/paths/benchmark.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from src.core.exceptions import AppError, ConfigError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.schemas import Split
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.path_score import normalized_adjacency, path_scores, shuffled_tails
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("benchmark_mkg_w_paths")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"
_MIN_LOGGED_PAIRS = 50


def _graph_scores(
    data: EntitySplit, adjacency: sparse.csr_matrix, gallery_ids: Sequence[int], max_hops: int, decay: float
) -> np.ndarray:
    """(queries × gallery) path score; zero rows for queries that name no anchor."""
    scores = np.zeros(data.clip_scores.shape, dtype=np.float32)
    anchors = sorted({q["anchor_id"] for q in data.queries if q["anchor_id"] is not None})
    per_anchor = path_scores(adjacency, anchors, max_hops, decay)[:, np.asarray(gallery_ids)]
    row_of = {anchor: row for row, anchor in enumerate(anchors)}
    for row, query in enumerate(data.queries):
        if query["anchor_id"] is not None:
            scores[row] = per_anchor[row_of[query["anchor_id"]]]
    return scores


def _select_on_val(
    val: EntitySplit, graph_by_decay: Mapping[float, np.ndarray], alphas: Sequence[float], top_ks: Sequence[int]
) -> dict[str, Any]:
    """Best (decay, alpha, top_k) on validation; see `select_alpha_top_k` for the order of preference."""
    candidates = [{"decay": decay, **select_alpha_top_k(val, graph, alphas, top_ks)}
                  for decay, graph in graph_by_decay.items()]
    return max(candidates, key=lambda c: (c["val_recall@1"], c["val_mrr"], c["alpha"], -c["top_k"]))


def run(config: MkgwRetrievalConfig) -> None:
    """Select the rerank setting on validation for both graphs, then report val and test."""
    data = MkgwConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    queries = read_json(processed / "queries.json")
    train_edges = [(head, tail) for head, _, tail in read_json(processed / "triples.json")["train"]]
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    gallery_ids = list(gallery_columns(entities))
    if max(config.top_k_grid) > len(gallery_ids):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery_ids)} entities")
    max_hops = data.rules.max_hops
    splits = {s: build_entity_split(s, entities, queries, image_table, query_table) for s in (Split.VAL, Split.TEST)}

    graphs = {
        "paths": train_edges,
        "paths_shuffled_edges": shuffled_tails(train_edges, config.path_shuffle_seed),
    }
    report: dict[str, Any] = {"model_name": config.clip.model_name, "gallery_entities": len(gallery_ids),
                              "max_hops": max_hops, "methods": {}}
    for method, edges in graphs.items():
        adjacency = normalized_adjacency(edges, len(entities))
        val_graphs = {decay: _graph_scores(splits[Split.VAL], adjacency, gallery_ids, max_hops, decay)
                      for decay in config.path_decay_grid}
        selected = _select_on_val(splits[Split.VAL], val_graphs, config.alpha_grid, config.top_k_grid)
        log.info("%s selected on val: %s", method, selected)
        report["methods"][method] = {"selected": selected, "splits": {}}
        for split, split_data in splits.items():
            graph = _graph_scores(split_data, adjacency, gallery_ids, max_hops, selected["decay"])
            fused = rerank_scores(split_data.clip_scores, graph, selected["alpha"], selected["top_k"])
            segments = segment_metrics(split_data, fused, relations, max_hops, config.recall_ks)
            # Graph alone over the whole gallery: what the paths know without any CLIP signal.
            graph_only = segment_metrics(split_data, graph, relations, max_hops, config.recall_ks)
            report["methods"][method]["splits"][split.value] = {
                "reranked": segments,
                "graph_only": {k: v for k, v in graph_only.items() if k.startswith(("group=indirect", "hops="))},
            }
            for name, row in segments.items():
                if row["answers"] >= _MIN_LOGGED_PAIRS and not name.startswith("relation="):
                    log.info("%-20s %-5s %-24s n=%5d R@1 %.3f R@10 %.3f MRR %.3f | graph only R@1 %s",
                             method, split.value, name, row["answers"], row["recall@1"], row["recall@10"],
                             row["mrr"], graph_only[name]["recall@1"] if name in report["methods"][method]["splits"][
                                 split.value]["graph_only"] else "-")
    out_path = config.runs_dir / "paths" / "benchmark.json"
    write_json(out_path, report)
    log.info("wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(MkgwRetrievalConfig.from_yaml(args.config))
    except AppError:
        log.exception("benchmark failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
