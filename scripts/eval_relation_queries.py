#!/usr/bin/env python3
"""Score CLIP and trained encoders on the relation queries and their objects-only twins.

    python -m scripts.build_relation_queries      # once
    python -m scripts.eval_relation_queries

For each method and seed, on the 1500 test images:
- full gallery: Recall@K and MRR, as in the benchmark;
- confusers: top-1 accuracy and MRR among the relevant image and its confusers only
  (images with the same two objects but without the triplet);
- masked (relation queries of trained encoders): the queried triplet is removed from the
  relevant image's graph before encoding. The drop from unmasked to masked is how much the
  model reads that edge; the graph holds the answer only in the unmasked setting.
Alpha and top_k are the ones each seed selected on the validation captions; nothing is tuned
here. Results are also split into generic predicates (on, in, has, ...) and the rest.
--runs-dir and --run score runs kept elsewhere (e.g. the main checkout).
Writes <runs_dir>/analysis/<out>.json (default relation_queries).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable, load_table
from src.evaluation.benchmark import SplitData, aggregate_runs, build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.evaluation.relation_queries import QueryKind, chance_accuracy, confuser_ranks, without_triplet
from src.models.graph_data import image_graph_to_data
from src.models.run_loading import GraphInputs, load_graph_inputs, load_trained_encoder
from src.models.train import encode_graphs
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("eval_relation_queries")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
QUERIES_DIR = "relation_queries"
# Predicates that hold between almost any two co-occurring objects; telling them apart
# rarely needs the image to show anything specific.
GENERIC_PREDICATES = frozenset({"on", "in", "has", "of", "with", "near", "by", "at", "next to", "and"})
# (method, run) pairs scored besides CLIP; the AdaMF runs isolate image-only and image + labels.
TRAINED = (
    ("graphsage", "directed-clip_image-l3-h512"),
    ("graphsage_label_text", "directed-clip_image-l3-h512-labeltext-d256"),
    ("adamf_mat", "adamf-d256-no-structure-no-text"),
    ("adamf_mat", "adamf-d256-no-structure"),
    ("adamf_mat", "adamf-d256"),
    ("adamf_mat", "adamf-mat-d256-mu1"),
)
_DECIMALS = 4


def _metrics(scores: np.ndarray, data: SplitData, confusers: Sequence[Sequence[int]], rows: np.ndarray,
             ks: Sequence[int]) -> dict[str, float]:
    """Full-gallery and confuser metrics of the given query rows."""
    full = retrieval_metrics(ranks_of_relevant(scores[rows], data.relevant_columns[rows]), ks)
    among = confuser_ranks(scores[rows], data.relevant_columns[rows], [confusers[r] for r in rows])
    return {**full, "confuser_acc@1": float((among == 1).mean()), "confuser_mrr": float((1.0 / among).mean())}


def _groups(queries: Sequence[dict[str, Any]], predicate_text: dict[int, str]) -> dict[str, np.ndarray]:
    """Row indices of all queries, generic-predicate queries and specific-predicate queries."""
    generic = np.asarray([predicate_text[q["triplet"][1]] in GENERIC_PREDICATES for q in queries], dtype=bool)
    return {"all": np.arange(len(queries)), "generic": np.flatnonzero(generic), "specific": np.flatnonzero(~generic)}


def main() -> int:
    """Score every method and seed on both query kinds, write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, help="read trained runs and benchmarks here instead")
    parser.add_argument("--run", action="append", metavar="METHOD:RUN", help="score these instead; repeatable")
    parser.add_argument("--out", default="relation_queries", help="report name under <runs_dir>/analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        runs_dir = args.runs_dir or config.runs_dir
        trained = [tuple(r.split(":", 1)) for r in args.run] if args.run else list(TRAINED)
        ks = config.benchmark.recall_ks
        processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
        all_queries = read_json(processed / QUERIES_DIR / "queries.json")
        table = load_table(config.model_embeddings_dir() / QUERIES_DIR, "queries")
        inputs: GraphInputs = load_graph_inputs(config)
        predicate_text = {
            int(p["predicate_id"]): p["text"] for p in read_json(processed / "graph" / "predicates.json")
        }
        graph_of = {int(g["vg_image_id"]): g for g in inputs.image_graphs}
        image_row = inputs.image_table.row_of()

        kinds = {}
        for kind in QueryKind:
            queries = sorted((q for q in all_queries if q["kind"] == kind.value), key=lambda q: q["query_id"])
            data = build_split_data(Split.TEST, inputs.images, queries, inputs.image_table, table)
            column = {image_id: c for c, image_id in enumerate(data.image_ids)}
            confusers = [[column[i] for i in q["confuser_ids"]] for q in queries]
            kinds[kind] = (queries, data, confusers, _groups(queries, predicate_text))

        # results[method][setting][kind][group] -> list of per-seed metric dicts
        results: dict[str, dict[str, dict[str, dict[str, list[dict[str, float]]]]]] = {}

        def record(method: str, setting: str, kind: QueryKind, scores: np.ndarray) -> None:
            _, data, confusers, groups = kinds[kind]
            slot = results.setdefault(method, {}).setdefault(setting, {}).setdefault(kind.value, {})
            for group, rows in groups.items():
                slot.setdefault(group, []).append(_metrics(scores, data, confusers, rows, ks))

        for kind in QueryKind:
            record("clip", "clip", kind, clip_scores(kinds[kind][1]))

        for method, run in trained:
            name = f"{method}/{run}"
            selected = read_json(runs_dir / "benchmark" / f"{method}-{run}.json")["seeds"]
            for seed in config.training.seeds:
                choice = selected[str(seed)]["selected_on_validation"]
                model = load_trained_encoder(runs_dir / method / run / f"seed{seed}", method, inputs)
                test_ids = kinds[QueryKind.RELATION][1].image_ids
                graphs = [image_graph_to_data(graph_of[i], image_row[i]) for i in test_ids]
                vectors = encode_graphs(model, graphs, config.training).numpy()
                graph_table = EmbeddingTable(ids=test_ids, vectors=vectors)
                for kind in QueryKind:
                    queries, data, _, _ = kinds[kind]
                    graph = graph_vector_scores(data, graph_table)
                    clip = clip_scores(data)
                    record(name, "graph_only", kind, graph)
                    record(name, "reranked", kind, rerank_scores(clip, graph, choice["alpha"], choice["top_k"]))
                    if kind is not QueryKind.RELATION:
                        continue
                    masked_graphs = [
                        image_graph_to_data(without_triplet(graph_of[q["relevant_ids"][0]], q["triplet"]),
                                            image_row[q["relevant_ids"][0]])
                        for q in queries
                    ]
                    masked_vectors = encode_graphs(model, masked_graphs, config.training).numpy()
                    masked = graph.copy()
                    rows = np.arange(len(queries))
                    masked[rows, data.relevant_columns] = (data.query_vectors * masked_vectors).sum(axis=1)
                    record(name, "graph_only_masked", kind, masked)
                    record(name, "reranked_masked", kind,
                           rerank_scores(clip, masked, choice["alpha"], choice["top_k"]))
                log.info("scored %s seed %d", name, seed)
        if not results:
            raise DataError("nothing scored")

        report = {
            "split": Split.TEST.value,
            "queries": {k.value: len(v[0]) for k, v in kinds.items()},
            "groups": {g: len(rows) for g, rows in kinds[QueryKind.RELATION][3].items()},
            "confuser_chance_acc@1": {
                g: round(chance_accuracy([kinds[QueryKind.RELATION][2][r] for r in rows]), _DECIMALS)
                for g, rows in kinds[QueryKind.RELATION][3].items()
            },
            "generic_predicates": sorted(GENERIC_PREDICATES),
            "methods": {
                method: {
                    setting: {
                        kind: {
                            group: {m: {k: round(v, _DECIMALS) for k, v in s.items()}
                                    for m, s in aggregate_runs(runs).items()}
                            for group, runs in groups.items()
                        }
                        for kind, groups in by_kind.items()
                    }
                    for setting, by_kind in settings.items()
                }
                for method, settings in results.items()
            },
        }
        out = config.runs_dir / "analysis" / f"{args.out}.json"
        write_json(out, report)
        log.info("wrote %s", out)
    except AppError:
        log.exception("relation query evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
