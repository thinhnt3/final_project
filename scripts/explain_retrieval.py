#!/usr/bin/env python3
"""Show the reranked results of caption queries with the relations behind each graph score.

Run from the repository root after `train_graphsage` and `run_benchmark --method graphsage`:
    python -m scripts.explain_retrieval --query-id coco-770337
    python -m scripts.explain_retrieval --sample 5              # random test queries

For every query: the rank of the correct image under CLIP and after reranking, then the top
results, each with its CLIP score, graph score, final score and the relations whose removal
changes the graph score most. Writes <runs_dir>/explanations/<run>-seed<seed>.json.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.benchmark import build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant
from src.models.ablations import ablate_graph
from src.models.checkpoint import GRAPH_VECTORS, RunDirectory
from src.models.explain import explain_graph_score
from src.models.graphsage import GraphImageEncoder
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("explain_retrieval")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
TRAINING_RUNS_SUBDIR = "graphsage"
BENCHMARK_SUBDIR = "benchmark"
OUTPUT_SUBDIR = "explanations"
GRAPH_DIR = "graph"
DEFAULT_RESULTS = 3
DEFAULT_RELATIONS = 3
_SCORE_DECIMALS = 4


def run(
    config: RetrievalConfig,
    seed: int,
    name: str,
    query_ids: list[str],
    sample: int,
    split: Split,
    n_results: int,
    n_relations: int,
) -> None:
    """Rerank the given queries with one trained seed and explain the top results."""
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    images, queries = read_json(processed / "images.json"), read_json(processed / "queries.json")
    image_table = load_table(embeddings_dir, "images")
    query_by_id = {q["query_id"]: q for q in queries}
    if not query_ids:
        pool = sorted(q["query_id"] for q in queries if q["split"] == split.value)
        query_ids = random.Random(seed).sample(pool, sample)
    unknown = [query_id for query_id in query_ids if query_id not in query_by_id]
    if unknown:
        raise DataError(f"unknown query ids: {unknown}")

    run_dir = RunDirectory(config.runs_dir / TRAINING_RUNS_SUBDIR / name / f"seed{seed}")

    benchmark_file = config.runs_dir / BENCHMARK_SUBDIR / f"graphsage-{name}.json"
    if not benchmark_file.is_file():
        raise DataError(f"{benchmark_file} not found; run the benchmark for this training run first")
    benchmark = read_json(benchmark_file)

    selected = benchmark["seeds"][str(seed)]["selected_on_validation"]
    alpha, top_k = float(selected["alpha"]), int(selected["top_k"])

    concepts = load_table(embeddings_dir, "concepts")
    predicates = load_table(embeddings_dir, "predicates")
    model = GraphImageEncoder(
        config.graphsage,
        torch.from_numpy(concepts.vectors),
        torch.from_numpy(predicates.vectors),
        torch.from_numpy(image_table.vectors),
    )
    model.load_state_dict(run_dir.load_best()["model"])
    graph_table = load_table(run_dir.path, GRAPH_VECTORS)
    graph_of = {int(g["vg_image_id"]): g for g in read_json(processed / GRAPH_DIR / "image_graphs.json")}
    concept_text = {c["concept_id"]: c["text"] for c in read_json(processed / GRAPH_DIR / "concepts.json")}
    predicate_text = {p["predicate_id"]: p["text"] for p in read_json(processed / GRAPH_DIR / "predicates.json")}
    image_row = image_table.row_of()

    split_data = {
        s: build_split_data(s, images, queries, image_table, load_table(embeddings_dir, "queries"))
        for s in {Split(query_by_id[query_id]["split"]) for query_id in query_ids}
    }
    report: list[dict[str, Any]] = []
    for query_id in query_ids:
        query = query_by_id[query_id]
        data = split_data[Split(query["split"])]
        row = data.query_ids.index(query_id)
        clip = clip_scores(data)[row : row + 1]
        graph = graph_vector_scores(data, graph_table)[row : row + 1]
        final = rerank_scores(clip, graph, alpha, min(top_k, len(data.image_ids)))
        relevant = data.relevant_columns[row : row + 1]
        clip_rank, final_rank = int(ranks_of_relevant(clip, relevant)[0]), int(ranks_of_relevant(final, relevant)[0])
        query_vector = torch.from_numpy(data.query_vectors[row])
        results = []
        for position, column in enumerate(np.argsort(-final[0])[:n_results], start=1):
            image_id = data.image_ids[column]
            _, contributions = explain_graph_score(
                model,
                ablate_graph(
                    graph_of[image_id],
                    config.graphsage.ablation,
                    config.graphsage.ablation_seed,
                ),
                image_row[image_id],
                query_vector,
                concept_text,
                predicate_text,
            )
            results.append(
                {
                    "rank": position,
                    "image_id": image_id,
                    "is_relevant": int(column) == int(relevant[0]),
                    "clip_score": round(float(clip[0, column]), _SCORE_DECIMALS),
                    "graph_score": round(float(graph[0, column]), _SCORE_DECIMALS),
                    "final_score": round(float(final[0, column]), _SCORE_DECIMALS),
                    "relations": [
                        {**c.to_dict(), "score_drop": round(c.score_drop, _SCORE_DECIMALS)}
                        for c in contributions[:n_relations]
                    ],
                }
            )
        report.append(
            {
                "query_id": query_id,
                "text": query["text"],
                "split": query["split"],
                "relevant_image_id": int(query["relevant_ids"][0]),
                "clip_rank": clip_rank,
                "reranked_rank": final_rank,
                "results": results,
            }
        )
        log.info(
            '"%s" (%s): correct image rank %d with CLIP, %d after reranking',
            query["text"],
            query_id,
            clip_rank,
            final_rank,
        )
        for result in results:
            relations = "; ".join(
                f"{r['subject']} {r['predicate']} {r['object']} ({r['score_drop']:+.3f})" for r in result["relations"]
            )
            log.info(
                "  #%d image %d%s clip %.3f graph %.3f final %.3f | %s",
                result["rank"],
                result["image_id"],
                " (correct)" if result["is_relevant"] else "",
                result["clip_score"],
                result["graph_score"],
                result["final_score"],
                relations or "no relations",
            )
    out_file = config.runs_dir / OUTPUT_SUBDIR / f"{name}-seed{seed}.json"
    write_json(out_file, {"run_name": name, "seed": seed, "alpha": alpha, "top_k": top_k, "queries": report})
    log.info("wrote %s", out_file)


def main() -> int:
    """Parse the command line and explain the requested queries."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="training run; default: the config's architecture")
    parser.add_argument("--seed", type=int, help="trained seed to use; default: the first seed of the config")
    parser.add_argument("--query-id", action="append", default=[], help="query to explain; repeatable")
    parser.add_argument("--sample", type=int, default=0, help="number of random queries when no --query-id is given")
    parser.add_argument("--split", choices=[s.value for s in Split], default=Split.TEST.value)
    parser.add_argument("--results", type=int, default=DEFAULT_RESULTS, help="top results shown per query")
    parser.add_argument("--relations", type=int, default=DEFAULT_RELATIONS, help="relations shown per result")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.query_id and args.sample < 1:
        parser.error("give --query-id or --sample N")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        run(
            config,
            args.seed if args.seed is not None else config.training.seeds[0],
            args.run or config.graphsage.run_name(),
            args.query_id,
            args.sample,
            Split(args.split),
            args.results,
            args.relations,
        )
    except AppError:
        log.exception("explanation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
