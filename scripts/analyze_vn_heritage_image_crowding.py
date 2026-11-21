#!/usr/bin/env python3
"""Why the graph helps on the place gallery but not on the image gallery of the VN heritage core.

Run from the repository root after `run_vn_heritage_retrieval --stage graphsage` on every split:
    python -m scripts.analyze_vn_heritage_image_crowding --split-seeds 1 2 3

All images of a place share the place's graph score. For every test (indirect query, answer
place) pair this counts the wrong images and the distinct wrong places ranked above the
answer's best image, under three scores:
    clip             CLIP cosine
    fused            alpha * CLIP + (1 - alpha) * graph, the reported method
    best_image_only  the same, but a place's graph score goes to its most query-similar image only
Written to <runs_dir>/image_crowding.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Any

import numpy as np

from scripts.run_vn_heritage_retrieval import Gallery, _load_splits, report_name
from src.core.exceptions import AppError
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.vn_heritage_query_config import QueryConfig
from src.embeddings.store import load_table
from src.evaluation.image_gallery import ImageSplit
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("analyze_vn_heritage_image_crowding")

DEFAULT_CONFIG = "configs/retrieval/vn_heritage_core.yaml"
_GRAPH = "claims"
_CUTOFF = 10


def wrong_above(test: ImageSplit, scores: np.ndarray) -> list[tuple[int, int]]:
    """(wrong images, distinct wrong places) above the best image of each indirect (query, answer) pair."""
    place = np.array(test.image_entities)
    counts = []
    for row, (groups, known) in enumerate(zip(test.answer_images, test.known, strict=True)):
        if test.queries[row]["group"] != "indirect":
            continue
        correct = np.zeros(scores.shape[1], dtype=bool)
        correct[[column for group in groups for column in group] + list(known)] = True
        for group in groups:
            above = (scores[row] >= scores[row, list(group)].max()) & ~correct
            counts.append((int(above.sum()), len(set(place[above].tolist()))))
    return counts


def best_image_mask(test: ImageSplit) -> np.ndarray:
    """True at each place's most CLIP-similar image, per query."""
    place = np.array(test.image_entities)
    best = np.zeros(test.clip_scores.shape, dtype=bool)
    for entity in np.unique(place):
        columns = np.flatnonzero(place == entity)
        best[np.arange(best.shape[0]), columns[test.clip_scores[:, columns].argmax(axis=1)]] = True
    return best


def analyze(base: MkgwRetrievalConfig, split_seeds: list[int]) -> dict[str, Any]:
    """Pool the counts of every claim split and training seed."""
    counts: dict[str, list[tuple[int, int]]] = {"clip": [], "fused": [], "best_image_only": []}
    for split_seed in split_seeds:
        name = f"split-{split_seed}"
        config = dataclasses.replace(base, embeddings_dir=base.embeddings_dir / name, runs_dir=base.runs_dir / name)
        data = QueryConfig.from_yaml(config.data_config).with_split_seed(split_seed)
        test = _load_splits(config, data, Gallery.IMAGE)[3][Split.TEST]
        if not isinstance(test, ImageSplit):
            raise AssertionError("the image gallery must give an ImageSplit")
        query_vectors = load_table(config.model_embeddings_dir(), "queries").select(
            [q["query_id"] for q in test.queries])
        run_dir = config.runs_dir / "graphsage" / _GRAPH
        best = best_image_mask(test)
        for seed, info in read_json(run_dir / report_name(Gallery.IMAGE))["seeds"].items():
            vectors = np.load(run_dir / f"seed-{seed}" / "entity_vectors.npy")
            graph = query_vectors @ vectors[list(test.image_entities)].T
            alpha, top_k = info["selected"]["alpha"], int(info["selected"]["top_k"])
            one_image = np.where(best, graph, graph.min(axis=1, keepdims=True))
            counts["clip"] += wrong_above(test, test.clip_scores.astype(np.float64))
            counts["fused"] += wrong_above(test, rerank_scores(test.clip_scores, graph, alpha, top_k))
            counts["best_image_only"] += wrong_above(test, rerank_scores(test.clip_scores, one_image, alpha, top_k))
    summary: dict[str, Any] = {"graph": _GRAPH, "claim_split_seeds": split_seeds, "scores": {}}
    for label, pairs in counts.items():
        table = np.array(pairs, dtype=float)
        summary["scores"][label] = {
            "pairs": len(pairs),
            "wrong_images_above": round(float(table[:, 0].mean()), 2),
            "wrong_places_above": round(float(table[:, 1].mean()), 2),
            "images_per_wrong_place_above": round(float(table[:, 0].sum() / table[:, 1].sum()), 2),
            f"answer_in_top_{_CUTOFF}_images": round(float((table[:, 0] < _CUTOFF).mean()), 4),
            f"answer_in_top_{_CUTOFF}_places": round(float((table[:, 1] < _CUTOFF).mean()), 4),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = MkgwRetrievalConfig.from_yaml(args.config)
        summary = analyze(config, args.split_seeds)
    except AppError:
        log.exception("analysis failed")
        return 1
    write_json(config.runs_dir / "image_crowding.json", summary)
    for label, row in summary["scores"].items():
        log.info("%-16s %s", label, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
