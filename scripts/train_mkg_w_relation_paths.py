#!/usr/bin/env python3
"""Train the relation path scorer on MKG-W and score it alone and fused with CLIP.

Run from the repository root after `embed_mkg_w`:
    python -m scripts.train_mkg_w_relation_paths --variant real
    python -m scripts.train_mkg_w_relation_paths --variant shuffled_edges

The scorer follows walks from the entity a relation query names; the weight of each step
depends on the relation and on the query text. The anchor is read from the query record
(`anchor_id`), i.e. entity linking is assumed solved. Queries without an anchor get a zero
graph score, so CLIP alone ranks them.

Each seed trains one model on the train relation queries; the best epoch is picked on the
validation relation queries, alpha and top-k on all of validation. Per-seed files go to
<runs_dir>/relation_paths/<variant>/seed-<n>/ and the summary over seeds to
<runs_dir>/relation_paths/<variant>/benchmark.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, ConfigError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable, load_table, verify_inputs_unchanged
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.models.kg_experiment import LOGGED_PREFIXES, Variant, seed_summary, train_query_set, variant_triples
from src.models.kg_train import KgQuerySet
from src.models.relation_paths import PathQuerySet, RelationPathScorer, fit_paths, gallery_path_scores
from src.models.train import seed_everything
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("train_mkg_w_relation_paths")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"


def _anchored(data: EntitySplit, query_table: EmbeddingTable) -> tuple[list[int], PathQuerySet]:
    """Rows of the split's queries that name an anchor, and those queries as a path query set."""
    rows = [row for row, q in enumerate(data.queries) if q["anchor_id"] is not None]
    vectors = torch.from_numpy(query_table.select([data.queries[row]["query_id"] for row in rows]))
    queries = KgQuerySet(vectors, [data.relevant[row] for row in rows], [data.known[row] for row in rows],
                         [[] for _ in rows])
    return rows, PathQuerySet(queries, torch.tensor([data.queries[row]["anchor_id"] for row in rows]))


def run(config: MkgwRetrievalConfig, variant: Variant) -> None:
    """Train one scorer per seed on the variant's graph, evaluate, and write the summary."""
    if variant is Variant.NO_EDGES:
        raise ConfigError("the path scorer has nothing to walk on without edges")
    if config.path_training.device.value != "cpu":
        raise ConfigError("the path scorer is trained on cpu so that runs match across machines")
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
    gallery = torch.tensor(list(column_of))
    if max(config.top_k_grid) > len(gallery):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery)} entities")
    graph = variant_triples(variant, read_json(processed / "triples.json")["train"], config.path_shuffle_seed)
    triples = torch.tensor(graph, dtype=torch.long)
    splits = {s: build_entity_split(s, entities, queries, image_table, query_table) for s in (Split.VAL, Split.TEST)}
    anchored = {s: _anchored(d, query_table) for s, d in splits.items()}
    train_records = sorted((q for q in queries if q["split"] == Split.TRAIN.value and q["anchor_id"] is not None),
                           key=lambda q: q["query_id"])
    train = PathQuerySet(
        train_query_set(train_records, query_table, column_of, {triple: row for row, triple in enumerate(graph)}),
        torch.tensor([q["anchor_id"] for q in train_records]),
    )
    log.info("variant=%s graph triples=%d train queries=%d val queries=%d gallery=%d", variant.value, len(graph),
             len(train.anchors), len(anchored[Split.VAL][0]), len(gallery))

    out_dir = config.runs_dir / "relation_paths" / variant.value
    max_hops = data.rules.max_hops
    seeds: dict[str, Any] = {}
    reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    for seed in config.path_training.seeds:
        log.info("seed %d", seed)
        seed_everything(seed)
        model = RelationPathScorer(config.relation_paths, triples, len(entities), len(relations),
                                   train.queries.query_vectors.shape[1])
        result = fit_paths(model, gallery, train, anchored[Split.VAL][1], config.path_training, seed,
                           config.recall_ks)
        model.load_state_dict(result.state_dict)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result.state_dict, seed_dir / "model.pt")
        write_json(seed_dir / "history.json", result.history)

        scores = {}
        for split, split_data in splits.items():
            rows, path_queries = anchored[split]
            scores[split] = np.zeros(split_data.clip_scores.shape, dtype=np.float32)
            scores[split][rows] = gallery_path_scores(model, path_queries, gallery)
        selected = select_alpha_top_k(splits[Split.VAL], scores[Split.VAL], config.alpha_grid, config.top_k_grid)
        seeds[str(seed)] = {"best_epoch": result.best_epoch, "stop_reason": result.stop_reason.value,
                            "selected": selected}
        log.info("seed %d best epoch %d, selected on val: %s", seed, result.best_epoch, selected)
        for split, split_data in splits.items():
            fused = rerank_scores(split_data.clip_scores, scores[split], selected["alpha"], int(selected["top_k"]))
            reranked[split].append(segment_metrics(split_data, fused, relations, max_hops, config.recall_ks))
            graph_only[split].append(segment_metrics(split_data, scores[split], relations, max_hops, config.recall_ks))

    report: dict[str, Any] = {
        "variant": variant.value,
        "model_name": config.clip.model_name,
        "graph_triples": len(graph),
        "max_hops": config.relation_paths.max_hops,
        "seeds": seeds,
        "splits": {s.value: {"reranked": seed_summary(reranked[s]), "graph_only": seed_summary(graph_only[s])}
                   for s in splits},
    }
    write_json(out_dir / "benchmark.json", report)
    for split in splits:
        for name, row in report["splits"][split.value]["reranked"].items():
            if name.startswith(LOGGED_PREFIXES):
                alone = report["splits"][split.value]["graph_only"][name]
                log.info("%-5s %-26s n=%5d rerank R@1 %.3f±%.3f R@10 %.3f MRR %.3f | graph only R@1 %.3f R@10 %.3f",
                         split.value, name, row["answers"], row["recall@1"]["mean"], row["recall@1"]["std"],
                         row["recall@10"]["mean"], row["mrr"]["mean"],
                         alone["recall@1"]["mean"], alone["recall@10"]["mean"])
    log.info("wrote %s", out_dir / "benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True, choices=[Variant.REAL.value, Variant.SHUFFLED_EDGES.value])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(MkgwRetrievalConfig.from_yaml(args.config), Variant(args.variant))
    except AppError:
        log.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
