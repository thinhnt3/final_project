#!/usr/bin/env python3
"""CLIP and CLIP + GraphSAGE retrieval on the VN heritage core.

Run from the repository root after `build_vn_heritage_queries`:
    python -m scripts.run_vn_heritage_retrieval --stage embed
    python -m scripts.run_vn_heritage_retrieval --stage clip
    python -m scripts.run_vn_heritage_retrieval --stage graphsage --graphs claims cue

A graph variant names the edges the encoder sees:
    no_edges         none: an MLP on each entity's own name vector
    claims           train claims only
    generic | kind | section | cue
                     train claims + article edges under that relation naming
    <name>_shuffled  the same edges with tails re-paired at random

Results go to <runs_dir>/clip/benchmark.json and <runs_dir>/graphsage/<graph>/benchmark.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.data.vn_heritage_query_config import QueryConfig
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import (
    META_FILE,
    EmbeddingTable,
    encode_in_batches,
    load_table,
    save_table,
    table_paths,
    verify_inputs_unchanged,
)
from src.evaluation.benchmark import aggregate_runs
from src.evaluation.metrics import retrieval_metrics
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.models.kg_graphsage import KgEntityEncoder
from src.models.kg_train import KgQuerySet, fit
from src.models.train import seed_everything
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.path_score import shuffled_tails
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("run_vn_heritage_retrieval")

DEFAULT_CONFIG = "configs/retrieval/vn_heritage_core.yaml"
_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
_NO_EDGES = "no_edges"
_CLAIMS = "claims"
_SHUFFLED = "_shuffled"
_LOGGED_PREFIXES = ("all", "group=", "hops=", "density=")

IdTriple = tuple[int, int, int]


def embed(config: MkgwRetrievalConfig, data: QueryConfig, image_dir: Path) -> None:
    """Encode image files, query texts and entity names; write the tables and their meta."""
    inputs = {"entities": data.out_dir / "entities.json", "queries": data.out_dir / "queries.json"}
    entities = read_json(inputs["entities"])
    queries = read_json(inputs["queries"])
    image_ids = [file for entity in entities for file in entity["image_files"]]
    missing = [file for file in image_ids if not (image_dir / file).is_file()]
    if missing:
        raise DataError(f"{len(missing)} image files are missing in {image_dir}, e.g. {missing[:3]}")

    encoder = ClipEncoder(config.clip.model_name, config.clip.device)
    log.info("model=%s device=%s dim=%d", config.clip.model_name, encoder.device, encoder.dim)
    text_batch = config.clip.text_batch_size
    jobs: dict[str, tuple[list[Any], list[Any], Any, int]] = {
        "entities": ([e["entity_id"] for e in entities], [e["name"] for e in entities],
                     encoder.encode_texts, text_batch),
        "queries": ([q["query_id"] for q in queries], [q["text"] for q in queries], encoder.encode_texts, text_batch),
        "images": (image_ids, [image_dir / file for file in image_ids], encoder.encode_image_files,
                   config.clip.image_batch_size),
    }
    out_dir = config.model_embeddings_dir()
    tables = {}
    for name, (ids, items, encode, batch_size) in jobs.items():
        tables[name] = EmbeddingTable(ids=tuple(ids), vectors=encode_in_batches(items, encode, batch_size))
        save_table(out_dir, name, tables[name])
        log.info("%s: %s", name, tables[name].vectors.shape)
    write_json(out_dir / META_FILE, {
        "model_name": config.clip.model_name,
        "device": encoder.device,
        "dim": encoder.dim,
        "tables": {
            name: {"rows": len(table.ids), "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)}}
            for name, table in tables.items()
        },
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs.values()},
    })
    log.info("wrote %s", out_dir)


def _load_splits(
    config: MkgwRetrievalConfig, data: QueryConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[Split, EntitySplit]]:
    """Entities, relations, queries and the val / test splits scored by CLIP."""
    entities = read_json(data.out_dir / "entities.json")
    relations = read_json(data.out_dir / "relations.json")
    queries = read_json(data.out_dir / "queries.json")
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    splits = {s: build_entity_split(s, entities, queries, image_table, query_table) for s in (Split.VAL, Split.TEST)}
    return entities, relations, queries, splits


def _segments(
    split_data: EntitySplit, scores: np.ndarray, relations: Sequence[Mapping[str, Any]], data: QueryConfig,
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Report segments of the shared benchmark plus one per density tier of the indirect queries."""
    segments = segment_metrics(split_data, scores, relations, data.rules.max_hops, ks)
    strata = read_json(data.out_dir / "strata.json")
    ranks, rows = split_data.ranks(scores)
    pairs_of: dict[str, list[int]] = {}
    for pair, row in enumerate(rows):
        query_id = split_data.queries[row]["query_id"]
        if query_id in strata:
            pairs_of.setdefault(f"density={strata[query_id]['tier']}", []).append(pair)
    for name, pairs in sorted(pairs_of.items()):
        metrics = retrieval_metrics(ranks[pairs], ks)
        segments[name] = {"answers": len(pairs), **{k: round(v, 4) for k, v in metrics.items()}}
    return segments


def run_clip(config: MkgwRetrievalConfig, data: QueryConfig) -> None:
    """Rank the gallery for every val and test query with CLIP cosine; write the report."""
    entities, relations, _, splits = _load_splits(config, data)
    ks = sorted({*config.recall_ks, *config.candidate_ks})
    report: dict[str, Any] = {"model_name": config.clip.model_name,
                              "gallery_entities": len(gallery_columns(entities)), "splits": {}}
    for split, split_data in splits.items():
        segments = _segments(split_data, split_data.clip_scores, relations, data, ks)
        report["splits"][split.value] = {"queries": len(split_data.queries), "segments": segments}
        for name, row in segments.items():
            if name.startswith(_LOGGED_PREFIXES):
                log.info("%-5s %-26s n=%4d R@1 %.3f R@5 %.3f R@10 %.3f MRR %.3f", split.value, name,
                         row["answers"], row["recall@1"], row["recall@5"], row["recall@10"], row["mrr"])
    out_path = config.runs_dir / "clip" / "benchmark.json"
    write_json(out_path, report)
    log.info("wrote %s", out_path)


def graph_triples(name: str, triples: Mapping[str, Any], seed: int) -> list[IdTriple]:
    """Edges of one graph variant; see the module docstring for the names."""
    base, shuffled = (name.removesuffix(_SHUFFLED), True) if name.endswith(_SHUFFLED) else (name, False)
    if base == _NO_EDGES:
        return []
    if base != _CLAIMS and base not in triples["articles"]:
        raise ConfigError(f"unknown graph {name!r}; article schemes are {sorted(triples['articles'])}")
    rows = triples[Split.TRAIN.value] + ([] if base == _CLAIMS else triples["articles"][base])
    graph = [(head, relation, tail) for head, relation, tail in rows]
    if shuffled:
        pairs = shuffled_tails([(head, tail) for head, _, tail in graph], seed)
        graph = [(head, relation, tail) for (head, tail), (_, relation, _) in zip(pairs, graph, strict=True)]
    return graph


def _train_queries(
    queries: Sequence[Mapping[str, Any]],
    query_table: EmbeddingTable,
    column_of: Mapping[int, int],
    triple_row: Mapping[IdTriple, int],
) -> KgQuerySet:
    """Train queries with the rows of the graph triples that state each query's own fact."""
    selected = sorted((q for q in queries if q["split"] == Split.TRAIN.value), key=lambda q: q["query_id"])
    own_triples = []
    for query in selected:
        rows = []
        for answer in query["relevant_ids"] if query["anchor_id"] is not None else ():
            head, tail = (query["anchor_id"], answer) if query["direction"] == "tail" else (answer, query["anchor_id"])
            # Absent in the shuffled and empty graphs: there the fact is not an edge.
            if (head, query["relation_id"], tail) in triple_row:
                rows.append(triple_row[(head, query["relation_id"], tail)])
        own_triples.append(rows)
    return KgQuerySet(
        query_vectors=torch.from_numpy(query_table.select([q["query_id"] for q in selected])),
        relevant=[[column_of[i] for i in q["relevant_ids"]] for q in selected],
        known=[[column_of[i] for i in q["known_ids"] if i in column_of] for q in selected],
        own_triples=own_triples,
    )


def _summary(per_seed: Sequence[Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """Mean and std over seeds of each metric, per report segment."""
    return {
        name: {"answers": per_seed[0][name]["answers"],
               **aggregate_runs([{m: seed[name][m] for m in _METRICS} for seed in per_seed])}
        for name in per_seed[0]
    }


def run_graphsage(config: MkgwRetrievalConfig, data: QueryConfig, graph_name: str) -> None:
    """Train one encoder per seed on one graph variant; score it alone and fused with CLIP."""
    if config.training.device.value != "cpu":
        raise ConfigError("the encoder is trained on cpu so that runs match across machines")
    entities, relations, queries, splits = _load_splits(config, data)
    embeddings_dir = config.model_embeddings_dir()
    query_table = load_table(embeddings_dir, "queries")
    entity_features = torch.from_numpy(load_table(embeddings_dir, "entities").select(list(range(len(entities)))))
    column_of = gallery_columns(entities)
    gallery = torch.tensor(list(column_of))
    if max(config.top_k_grid) > len(gallery):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery)} entities")

    graph = graph_triples(graph_name, read_json(data.out_dir / "triples.json"), config.path_shuffle_seed)
    triples = torch.tensor(graph, dtype=torch.long).reshape(-1, 3)
    train = _train_queries(queries, query_table, column_of, {triple: row for row, triple in enumerate(graph)})
    val_data = splits[Split.VAL]
    val = KgQuerySet(torch.from_numpy(query_table.select([q["query_id"] for q in val_data.queries])),
                     val_data.relevant, val_data.known, [[] for _ in val_data.queries])
    log.info("graph=%s triples=%d train queries=%d (own fact removed for %d) gallery=%d", graph_name, len(graph),
             len(train.relevant), sum(bool(rows) for rows in train.own_triples), len(gallery))

    out_dir = config.runs_dir / "graphsage" / graph_name
    seeds: dict[str, Any] = {}
    reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    for seed in config.training.seeds:
        seed_everything(seed)  # weights are drawn when the model is built
        model = KgEntityEncoder(config.graphsage, entity_features, len(relations))
        result = fit(model, triples, gallery, train, val, config.training, seed, config.recall_ks)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "entity_vectors.npy", result.entity_vectors)
        write_json(seed_dir / "history.json", result.history)
        gallery_vectors = result.entity_vectors[gallery.numpy()]
        scores = {s: query_table.select([q["query_id"] for q in d.queries]) @ gallery_vectors.T
                  for s, d in splits.items()}
        selected = select_alpha_top_k(val_data, scores[Split.VAL], config.alpha_grid, config.top_k_grid)
        seeds[str(seed)] = {"best_epoch": result.best_epoch, "stop_reason": result.stop_reason.value,
                            "selected": selected}
        log.info("seed %d best epoch %d, selected on val: %s", seed, result.best_epoch, selected)
        for split, split_data in splits.items():
            fused = rerank_scores(split_data.clip_scores, scores[split], selected["alpha"], int(selected["top_k"]))
            reranked[split].append(_segments(split_data, fused, relations, data, config.recall_ks))
            graph_only[split].append(_segments(split_data, scores[split], relations, data, config.recall_ks))

    report: dict[str, Any] = {
        "graph": graph_name,
        "model_name": config.clip.model_name,
        "graph_triples": len(graph),
        "seeds": seeds,
        "splits": {s.value: {"reranked": _summary(reranked[s]), "graph_only": _summary(graph_only[s])}
                   for s in splits},
    }
    write_json(out_dir / "benchmark.json", report)
    for split in splits:
        for name, row in report["splits"][split.value]["reranked"].items():
            if name.startswith(_LOGGED_PREFIXES):
                alone = report["splits"][split.value]["graph_only"][name]
                log.info("%-5s %-26s n=%4d rerank R@1 %.3f±%.3f R@10 %.3f MRR %.3f | graph only R@1 %.3f R@10 %.3f",
                         split.value, name, row["answers"], row["recall@1"]["mean"], row["recall@1"]["std"],
                         row["recall@10"]["mean"], row["mrr"]["mean"],
                         alone["recall@1"]["mean"], alone["recall@10"]["mean"])
    log.info("wrote %s", out_dir / "benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True, choices=["embed", "clip", "graphsage"])
    parser.add_argument("--graphs", nargs="+", default=[_CLAIMS], help="graph variants for --stage graphsage")
    parser.add_argument("--image-dir", help="override paths.image_dir of the query config")
    parser.add_argument("--split-seed", type=int, help="use the task built with the same --split-seed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = MkgwRetrievalConfig.from_yaml(args.config)
        data = QueryConfig.from_yaml(config.data_config)
        if args.split_seed is not None:
            data = data.with_split_seed(args.split_seed)
            name = f"split-{args.split_seed}"
            config = dataclasses.replace(
                config, embeddings_dir=config.embeddings_dir / name, runs_dir=config.runs_dir / name
            )
        if args.stage == "embed":
            embed(config, data, Path(args.image_dir) if args.image_dir else data.image_dir)
        elif args.stage == "clip":
            run_clip(config, data)
        else:
            for graph_name in args.graphs:
                run_graphsage(config, data, graph_name)
    except AppError:
        log.exception("%s failed", args.stage)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
