#!/usr/bin/env python3
"""Train the GraphSAGE entity encoder on MKG-W and score it alone and fused with CLIP.

Run from the repository root after `embed_mkg_w`:
    python -m scripts.train_mkg_w_graphsage --variant real
    python -m scripts.train_mkg_w_graphsage --variant shuffled_edges
    python -m scripts.train_mkg_w_graphsage --variant no_edges

Variants differ only in the graph the encoder sees:
    real            the train triples
    shuffled_edges  the same heads, relations and tails, with tails re-paired at random
    no_edges        no triples: the encoder is an MLP on each entity's own name vector

Each seed trains one model. The best epoch is picked on the validation queries named by
`kg_graphsage.selection_queries`; alpha and top-k are picked on all of validation, per seed.
`--structure-only-rate` overrides `kg_graphsage.structure_only_rate` for the run.

Per-seed files go to <runs_dir>/graphsage/<run>/seed-<n>/ and the summary over seeds to
<runs_dir>/graphsage/<run>/benchmark.json, where <run> is the variant plus `-so<rate>` when
structure-only rows are on and `-sel-<queries>` when the epoch is not picked on all queries.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, ConfigError, DataError
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
from src.models.kg_graphsage import KgEntityEncoder
from src.models.kg_train import KgQuerySet, fit
from src.models.train import seed_everything
from src.retrieval.mkg_w_config import MkgwRetrievalConfig, SelectionQueries
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("train_mkg_w_graphsage")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"


def _eval_queries(data: EntitySplit, query_table: EmbeddingTable, selection: SelectionQueries) -> KgQuerySet:
    """Validation queries that pick the best epoch."""
    rows = [row for row, q in enumerate(data.queries)
            if selection is SelectionQueries.ALL or q["group"] == selection.value]
    vectors = torch.from_numpy(query_table.select([data.queries[row]["query_id"] for row in rows]))
    return KgQuerySet(vectors, [data.relevant[row] for row in rows], [data.known[row] for row in rows],
                      [[] for _ in rows])


def run_name(variant: Variant, config: MkgwRetrievalConfig) -> str:
    """Directory name of a run: the variant plus every setting that differs from the plain run."""
    name = variant.value
    if config.graphsage.structure_only_rate > 0:
        name += f"-so{config.graphsage.structure_only_rate:g}"
    if config.selection_queries is not SelectionQueries.ALL:
        name += f"-sel-{config.selection_queries.value}"
    return name


def run(config: MkgwRetrievalConfig, variant: Variant) -> None:
    """Train one model per seed on the variant's graph, evaluate, and write the summary."""
    if config.training.device.value != "cpu":
        raise ConfigError("the MKG-W encoder is trained on cpu so that runs match across machines")
    data = MkgwConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    queries = read_json(processed / "queries.json")
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    entity_features = torch.from_numpy(load_table(embeddings_dir, "entities").select(list(range(len(entities)))))

    column_of = gallery_columns(entities)
    gallery = torch.tensor(list(column_of))
    if max(config.top_k_grid) > len(gallery):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery)} entities")
    graph = variant_triples(variant, read_json(processed / "triples.json")["train"], config.path_shuffle_seed)
    triples = torch.tensor(graph, dtype=torch.long).reshape(-1, 3)
    splits = {s: build_entity_split(s, entities, queries, image_table, query_table) for s in (Split.VAL, Split.TEST)}
    train = train_query_set(queries, query_table, column_of, {triple: row for row, triple in enumerate(graph)})
    val = _eval_queries(splits[Split.VAL], query_table, config.selection_queries)
    masked = sum(bool(rows) for rows in train.own_triples)
    log.info("run=%s graph triples=%d train queries=%d (own fact removed for %d) selection queries=%d gallery=%d",
             run_name(variant, config), len(graph), len(train.relevant), masked, len(val.relevant), len(gallery))
    if variant is Variant.REAL and masked == 0:
        raise DataError("no train query matched its own triple; the graph and the queries are out of sync")

    out_dir = config.runs_dir / "graphsage" / run_name(variant, config)
    max_hops = data.rules.max_hops
    seeds: dict[str, Any] = {}
    reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    for seed in config.training.seeds:
        log.info("seed %d", seed)
        seed_everything(seed)  # weights are drawn when the model is built
        model = KgEntityEncoder(config.graphsage, entity_features, len(relations))
        result = fit(model, triples, gallery, train, val, config.training, seed, config.recall_ks)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "entity_vectors.npy", result.entity_vectors)
        torch.save(result.state_dict, seed_dir / "model.pt")
        write_json(seed_dir / "history.json", result.history)

        gallery_vectors = result.entity_vectors[gallery.numpy()]
        scores = {s: query_table.select([q["query_id"] for q in d.queries]) @ gallery_vectors.T
                  for s, d in splits.items()}
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
        "structure_only_rate": config.graphsage.structure_only_rate,
        "selection_queries": config.selection_queries.value,
        "model_name": config.clip.model_name,
        "graph_triples": len(graph),
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
    parser.add_argument("--variant", required=True, choices=[v.value for v in Variant])
    parser.add_argument("--structure-only-rate", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = MkgwRetrievalConfig.from_yaml(args.config)
        if args.structure_only_rate is not None:
            config = replace(config, graphsage=replace(config.graphsage, structure_only_rate=args.structure_only_rate))
        run(config, Variant(args.variant))
    except AppError:
        log.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
