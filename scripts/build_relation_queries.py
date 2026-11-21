#!/usr/bin/env python3
"""Build the relation queries of the test split and embed them with the frozen CLIP.

    python -m scripts.build_relation_queries                 # test split
    python -m scripts.build_relation_queries --split val     # for validation during training

Writes <processed>/relation_queries/<name>.json (see src/evaluation/relation_queries.py
for the selection rule and the templates) and the CLIP table
<embeddings>/<model>/relation_queries/<name>.{npy,ids.json} with <name>.meta.json, where
<name> is `queries` for test and `queries_<split>` otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import EmbeddingTable, encode_in_batches, save_table, table_paths
from src.evaluation.relation_queries import TEMPLATES, QueryKind, build_relation_queries
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("build_relation_queries")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
QUERY_SEED = 13
QUERIES_DIR = "relation_queries"


def table_name(split: Split) -> str:
    """File and table name of a split's relation queries; test keeps the original name."""
    return "queries" if split is Split.TEST else f"queries_{split.value}"


def main() -> int:
    """Build, embed and write the relation queries."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=[Split.VAL.value, Split.TEST.value], default=Split.TEST.value)
    args = parser.parse_args()
    split = Split(args.split)
    name = table_name(split)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
        graph_file = processed / "graph" / "image_graphs.json"
        concepts = {int(c["concept_id"]): c["text"] for c in read_json(processed / "graph" / "concepts.json")}
        predicates = {int(p["predicate_id"]): p["text"] for p in read_json(processed / "graph" / "predicates.json")}
        queries = build_relation_queries(read_json(graph_file), concepts, predicates, QUERY_SEED, split)
        relation = [q for q in queries if q["kind"] == QueryKind.RELATION.value]
        log.info("%d relation queries (+ as many objects twins); top predicates %s", len(relation),
                 Counter(predicates[q["triplet"][1]] for q in relation).most_common(8))

        out_file = processed / QUERIES_DIR / f"{name}.json"
        write_json(out_file, queries)
        encoder = ClipEncoder(config.clip.model_name, config.clip.device)
        vectors = encode_in_batches([q["text"] for q in queries], encoder.encode_texts, config.clip.text_batch_size)
        table = EmbeddingTable(ids=tuple(q["query_id"] for q in queries), vectors=vectors)
        out_dir = config.model_embeddings_dir() / QUERIES_DIR
        save_table(out_dir, name, table)
        write_json(out_dir / f"{name}.meta.json", {
            "model_name": config.clip.model_name,
            "seed": QUERY_SEED,
            "templates": {str(k): v for k, v in TEMPLATES.items()},
            "inputs_sha256": {str(p): sha256_of(p) for p in (graph_file, out_file)},
            "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)},
        })
        log.info("wrote %s and %s", out_file, out_dir)
    except AppError:
        log.exception("building relation queries failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
