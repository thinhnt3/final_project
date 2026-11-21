#!/usr/bin/env python3
"""Score the already-selected MKG-W methods on the extra test sets, with given and with linked anchors.

Run from the repository root after `build_mkg_w_extra_queries`, `benchmark_mkg_w_paths` and
`train_mkg_w_relation_paths --variant real`:
    python -m scripts.benchmark_mkg_w_extra_queries

Nothing is trained or selected here. The path score keeps the decay, alpha and top-k it got on
validation; each relation path scorer seed keeps its trained weights, alpha and top-k.

Query sets: `test` (the regular test queries), `chain` and `handwritten` (see
`build_mkg_w_extra_queries`). Anchor sources: `given` reads the anchors stored in the query
record; `linked` finds them in the query text with `AnchorLinker` and applies to every query,
including name and description queries, because a deployed system does not know the group.
A query with several anchors gets the sum of its per-anchor graph scores.

`linked_tuned` is the one place where something is selected: with linked anchors the named
entity itself may be the answer (a name query), so it gets a graph score too. That score
(`anchor_linking.self_weight_grid`), alpha and top-k are chosen on all validation queries with
linked anchors, then applied unchanged to every set.

Writes <runs_dir>/extra_queries/benchmark.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, DataError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.mkg_w_queries import EXTRA_QUERIES_FILE
from src.data.mkg_w_raw import display_name
from src.data.schemas import Split
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import EmbeddingTable, encode_in_batches, load_table, verify_inputs_unchanged
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.models.kg_experiment import seed_summary
from src.models.relation_paths import RelationPathScorer
from src.retrieval.anchor_linking import AnchorLinker
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.path_score import normalized_adjacency, path_scores
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("benchmark_mkg_w_extra_queries")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"
_PAIR_BATCH = 256

# (query rows, anchor ids), one entry per (query, anchor) pair → (pairs × gallery) scores.
PairScores = Callable[[np.ndarray, np.ndarray], np.ndarray]
Segments = dict[str, dict[str, float]]


def given_anchors(query: Mapping[str, Any]) -> list[int]:
    """Anchors stored in the query record: `anchor_ids`, or the single `anchor_id` of a relation query."""
    if "anchor_ids" in query:
        return list(query["anchor_ids"])
    return [] if query["anchor_id"] is None else [query["anchor_id"]]


def linking_stats(given: Sequence[Sequence[int]], linked: Sequence[Sequence[int]]) -> dict[str, float]:
    """How the linked anchors compare with the given ones, as shares of the queries that have given anchors."""
    pairs = [(set(g), set(found)) for g, found in zip(given, linked, strict=True) if g]
    if not pairs:
        return {}
    return {
        "queries_with_given_anchors": len(pairs),
        "same_anchors": round(sum(g == found for g, found in pairs) / len(pairs), 4),
        "all_given_found": round(sum(g <= found for g, found in pairs) / len(pairs), 4),
        "none_found": round(sum(not found for _, found in pairs) / len(pairs), 4),
    }


def pooled_scores(anchors: Sequence[Sequence[int]], n_gallery: int, pair_scores: PairScores) -> np.ndarray:
    """(queries × gallery) graph score: per-anchor scores summed, each row divided by its largest value.

    A query without anchors keeps a zero row, so CLIP alone ranks it.
    """
    rows = np.array([row for row, ids in enumerate(anchors) for _ in ids], dtype=np.int64)
    ids = np.array([anchor for per_query in anchors for anchor in per_query], dtype=np.int64)
    scores = np.zeros((len(anchors), n_gallery), dtype=np.float32)
    for start in range(0, len(rows), _PAIR_BATCH):
        batch = slice(start, start + _PAIR_BATCH)
        np.add.at(scores, rows[batch], pair_scores(rows[batch], ids[batch]))
    peak = scores.max(axis=1, keepdims=True)
    return np.divide(scores, peak, out=np.zeros_like(scores), where=peak > 0)


def with_self_score(graph: np.ndarray, anchor_columns: Sequence[Sequence[int]], weight: float) -> np.ndarray:
    """Add `weight` to each query's own anchors (gallery columns) and rescale rows to a largest value of 1."""
    if weight == 0:
        return graph
    scores = graph.copy()
    for row, columns in enumerate(anchor_columns):
        scores[row, list(columns)] += weight
    peak = scores.max(axis=1, keepdims=True)
    return np.divide(scores, peak, out=np.zeros_like(scores), where=peak > 0)


def select_with_self_score(val: EntitySplit, graph: np.ndarray, anchor_columns: Sequence[Sequence[int]],
                           config: MkgwRetrievalConfig) -> dict[str, float]:
    """Best (self weight, alpha, top-k) on validation; a tie goes to the smaller self weight."""
    candidates = [{"self_weight": weight, **select_alpha_top_k(
        val, with_self_score(graph, anchor_columns, weight), config.alpha_grid, config.top_k_grid)}
        for weight in config.anchor_self_weight_grid]
    return max(candidates, key=lambda c: (c["val_recall@1"], c["val_mrr"], -c["self_weight"]))


def run(config: MkgwRetrievalConfig) -> None:
    """Score CLIP, the path score and the relation path scorer on every query set."""
    data = MkgwConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    train_triples = read_json(processed / "triples.json")["train"]
    extra = read_json(processed / EXTRA_QUERIES_FILE)
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    column_of = gallery_columns(entities)
    gallery_ids = np.array(list(column_of))
    max_hops = data.rules.max_hops

    encoder = ClipEncoder(config.clip.model_name, config.clip.device)
    sets: dict[str, tuple[EntitySplit, np.ndarray]] = {}
    regular = load_table(embeddings_dir, "queries")
    regular_queries = read_json(processed / "queries.json")
    # Validation comes first: it only selects the `linked_tuned` setting and is not a reported set.
    for name, part in (("val", Split.VAL), ("test", Split.TEST)):
        split = build_entity_split(part, entities, regular_queries, image_table, regular)
        sets[name] = (split, regular.select([q["query_id"] for q in split.queries]))
    for name, records in extra.items():
        encoded = encode_in_batches([q["text"] for q in records], encoder.encode_texts, config.clip.text_batch_size)
        table = EmbeddingTable(ids=tuple(q["query_id"] for q in records), vectors=encoded)
        split = build_entity_split(Split.TEST, entities, records, image_table, table)
        sets[name] = (split, table.select([q["query_id"] for q in split.queries]))
        log.info("%s: %d queries encoded", name, len(records))

    linker = AnchorLinker([display_name(entity["name"]) for entity in entities])
    paths_selected = read_json(config.runs_dir / "paths" / "benchmark.json")["methods"]["paths"]["selected"]
    adjacency = normalized_adjacency([(head, tail) for head, _, tail in train_triples], len(entities))
    scorer_report = read_json(config.runs_dir / "relation_paths" / "real" / "benchmark.json")
    triples = torch.tensor(train_triples, dtype=torch.long)
    scorers: dict[str, RelationPathScorer] = {}
    for seed in scorer_report["seeds"]:
        model = RelationPathScorer(config.relation_paths, triples, len(entities), len(relations), encoder.dim)
        model.load_state_dict(torch.load(config.runs_dir / "relation_paths" / "real" / f"seed-{seed}" / "model.pt"))
        scorers[seed] = model.eval()

    report: dict[str, Any] = {"model_name": config.clip.model_name, "path_score_selected": paths_selected,
                              "relation_paths_selected": {s: v["selected"] for s, v in scorer_report["seeds"].items()},
                              "linked_tuned_selected": {}, "sets": {}}
    tuned: dict[str, dict[str, float]] = report["linked_tuned_selected"]
    for name, (split, query_vectors) in sets.items():
        def metrics(scores: np.ndarray, split: EntitySplit = split) -> Segments:
            return segment_metrics(split, scores, relations, max_hops, config.recall_ks)

        vectors = torch.from_numpy(query_vectors)
        anchors = {"given": [given_anchors(q) for q in split.queries],
                   "linked": [linker.link(q["text"]) for q in split.queries]}
        methods: dict[str, Any] = {"clip": metrics(split.clip_scores)}
        own_columns = [[column_of[a] for a in ids if a in column_of] for ids in anchors["linked"]]
        for source, per_query in anchors.items():
            if any(anchor >= len(entities) for ids in per_query for anchor in ids):
                raise DataError(f"{name}: an anchor is not an entity id")
            walk = pooled_scores(per_query, len(gallery_ids), lambda _rows, ids: path_scores(
                adjacency, ids.tolist(), max_hops, paths_selected["decay"])[:, gallery_ids])
            methods[f"path_score/{source}"] = {
                "reranked": metrics(rerank_scores(split.clip_scores, walk, paths_selected["alpha"],
                                                  paths_selected["top_k"])),
                "graph_only": metrics(walk),
            }
            if source == "linked":
                if name == "val":
                    tuned["path_score"] = select_with_self_score(split, walk, own_columns, config)
                chosen = tuned["path_score"]
                methods["path_score/linked_tuned"] = {"reranked": metrics(rerank_scores(
                    split.clip_scores, with_self_score(walk, own_columns, chosen["self_weight"]),
                    chosen["alpha"], int(chosen["top_k"])))}
            reranked, alone, retuned = [], [], []
            for seed, model in scorers.items():
                def learned(rows: np.ndarray, ids: np.ndarray, model: RelationPathScorer = model,
                            vectors: torch.Tensor = vectors) -> np.ndarray:
                    with torch.inference_mode():
                        return model(vectors[rows], torch.from_numpy(ids))[:, gallery_ids].numpy()

                graph = pooled_scores(per_query, len(gallery_ids), learned)
                selected = scorer_report["seeds"][seed]["selected"]
                reranked.append(metrics(rerank_scores(split.clip_scores, graph, selected["alpha"],
                                                      int(selected["top_k"]))))
                alone.append(metrics(graph))
                if source == "linked":
                    if name == "val":
                        tuned[f"relation_paths/{seed}"] = select_with_self_score(split, graph, own_columns, config)
                    chosen = tuned[f"relation_paths/{seed}"]
                    retuned.append(metrics(rerank_scores(
                        split.clip_scores, with_self_score(graph, own_columns, chosen["self_weight"]),
                        chosen["alpha"], int(chosen["top_k"]))))
            methods[f"relation_paths/{source}"] = {"reranked": seed_summary(reranked),
                                                   "graph_only": seed_summary(alone)}
            if retuned:
                methods["relation_paths/linked_tuned"] = {"reranked": seed_summary(retuned)}
        if name == "val":
            log.info("linked_tuned selected on val: %s", tuned)
            continue
        report["sets"][name] = {"queries": len(split.queries), "linking": linking_stats(anchors["given"],
                                                                                         anchors["linked"]),
                                "methods": methods}
        log.info("%s linking: %s", name, report["sets"][name]["linking"])
        for segment, row in methods["clip"].items():
            if segment.startswith("relation="):
                continue
            line = [f"clip {row['recall@1']:.3f}"]
            for source in anchors:
                line.append(f"path/{source} {methods[f'path_score/{source}']['reranked'][segment]['recall@1']:.3f}")
                learned_row = methods[f"relation_paths/{source}"]["reranked"][segment]["recall@1"]
                line.append(f"scorer/{source} {learned_row['mean']:.3f}±{learned_row['std']:.3f}")
            retuned_row = methods["relation_paths/linked_tuned"]["reranked"][segment]["recall@1"]
            line.append(f"path/tuned {methods['path_score/linked_tuned']['reranked'][segment]['recall@1']:.3f}")
            line.append(f"scorer/tuned {retuned_row['mean']:.3f}±{retuned_row['std']:.3f}")
            log.info("%-11s %-28s n=%5d R@1 %s", name, segment, row["answers"], " | ".join(line))

    out_path = config.runs_dir / "extra_queries" / "benchmark.json"
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
