#!/usr/bin/env python3
"""Score frozen CLIP on MKG-W: text query → entity images, ranked per entity.

Run from the repository root after `embed_mkg_w`:
    python -m scripts.benchmark_mkg_w_clip

The gallery is every entity that has images. Results are reported for val and test, split
by query group and kind, by hop distance and by relation direction, and written to
<runs_dir>/clip/benchmark.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np

from src.core.exceptions import AppError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.schemas import Split
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.mkg_w_benchmark import build_entity_split, gallery_columns, segment_metrics
from src.retrieval.mkg_w_config import MkgwRetrievalConfig

log = logging.getLogger("benchmark_mkg_w_clip")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"
# Segments with fewer scored answers are too noisy to print; they stay in the JSON.
_MIN_LOGGED_PAIRS = 50


def run(config: MkgwRetrievalConfig) -> None:
    """Rank the gallery for every val and test query with CLIP cosine; write the report."""
    data = MkgwConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    queries = read_json(processed / "queries.json")
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    column_of = gallery_columns(entities)
    ks = sorted({*config.recall_ks, *config.candidate_ks})

    report: dict[str, Any] = {"model_name": config.clip.model_name, "gallery_entities": len(column_of), "splits": {}}
    for split in (Split.VAL, Split.TEST):
        split_data = build_entity_split(split, entities, queries, image_table, query_table)
        segments = segment_metrics(split_data, split_data.clip_scores, relations, data.rules.max_hops, ks)
        # How often CLIP's first choice is the entity named in the query instead of the answer.
        top = split_data.clip_scores.argmax(axis=1)
        anchored = [(row, column_of[q["anchor_id"]]) for row, q in enumerate(split_data.queries)
                    if q["anchor_id"] is not None and q["anchor_id"] in column_of]
        anchor_first = round(float(np.mean([top[row] == column for row, column in anchored])), 4)
        report["splits"][split.value] = {
            "queries": len(split_data.queries),
            "segments": segments,
            "indirect_anchor_in_gallery": len(anchored),
            "indirect_anchor_ranked_first": anchor_first,
        }
        for name, row in segments.items():
            if row["answers"] >= _MIN_LOGGED_PAIRS and not name.startswith("relation="):
                log.info("%-5s %-28s n=%5d R@1 %.3f R@10 %.3f MRR %.3f | in top-100 %.3f top-1000 %.3f",
                         split.value, name, row["answers"], row["recall@1"], row["recall@10"], row["mrr"],
                         row.get("recall@100", float("nan")), row.get("recall@1000", float("nan")))
        log.info("%-5s anchor ranked first: %.3f of %d indirect queries whose anchor has images",
                 split.value, anchor_first, len(anchored))
    out_path = config.runs_dir / "clip" / "benchmark.json"
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
