#!/usr/bin/env python3
"""Audit existing ablation predictions without tuning or retraining.

python -m scripts.analyze_ablation
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.ablation_analysis import paired_analysis
from src.evaluation.benchmark import build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.models.ablations import Ablation
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores


def run(
    config_path: str,
    output: str,
) -> None:
    """Keep official alpha/k; fixed-k=500 is explicitly a post-hoc diagnostic."""
    # Load the held-out split without changing the retrieval configuration.
    config = RetrievalConfig.from_yaml(config_path)
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    embeddings = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings)
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")
    data = build_split_data(
        Split.TEST,
        images,
        queries,
        load_table(embeddings, "images"),
        load_table(embeddings, "queries"),
    )
    clip = clip_scores(data)
    inputs = [Path(config_path), embeddings / "meta.json", processed / "graph/image_graphs.json"]
    official: dict[str, Any] = {}
    fixed_k: dict[str, Any] = {}
    ranks: dict[str, np.ndarray] = {}

    # Reconstruct each trained variant using its original validation selection.
    for variant in Ablation:
        cfg = replace(config, graphsage=replace(config.graphsage, ablation=variant))
        name = cfg.graphsage.run_name()
        benchmark_path = config.runs_dir / "benchmark" / f"graphsage-{name}.json"
        inputs.append(benchmark_path)
        benchmark = read_json(benchmark_path)
        validate_artifact_inputs(benchmark, str(benchmark_path))
        official[variant.value] = {}
        fixed_k[variant.value] = {}
        per_seed_ranks = []
        for seed in config.training.seeds:
            path = config.runs_dir / "graphsage" / name / f"seed{seed}"
            validate_run(
                path,
                cfg,
                seed,
            )
            inputs.extend([path / "config.yaml", path / "graph_vectors.npy"])
            graph = graph_vector_scores(data, load_table(path, "graph_vectors"))
            selected = benchmark["seeds"][str(seed)]["selected_on_validation"]
            rank = ranks_of_relevant(
                rerank_scores(
                    clip,
                    graph,
                    selected["alpha"],
                    selected["top_k"],
                ),
                data.relevant_columns,
            )
            per_seed_ranks.append(rank)
            official[variant.value][str(seed)] = {"selection": selected, "metrics": retrieval_metrics(rank)}

            # Fixed k=500 is a post-hoc diagnostic, not a new tuned result.
            diagnostic = ranks_of_relevant(
                rerank_scores(
                    clip,
                    graph,
                    selected["alpha"],
                    min(500, len(data.image_ids)),
                ),
                data.relevant_columns,
            )
            fixed_k[variant.value][str(seed)] = retrieval_metrics(diagnostic)
        ranks[variant.value] = np.stack(per_seed_ranks)

    # Keep captions of the same true image together in the bootstrap.
    true_images = np.array(data.image_ids)[data.relevant_columns]
    comparisons = {
        v: paired_analysis(
            ranks["full"],
            values,
            true_images,
        )
        for v, values in ranks.items()
        if v != "full"
    }

    # Collect ten descriptive wins/losses without assigning query-type labels.
    query_of = {q["query_id"]: q for q in queries}
    graph_of = {int(g["vg_image_id"]): g for g in read_json(processed / "graph/image_graphs.json")}
    concepts = {c["concept_id"]: c["text"] for c in read_json(processed / "graph/concepts.json")}
    predicates = {p["predicate_id"]: p["text"] for p in read_json(processed / "graph/predicates.json")}
    cases = []
    full = ranks["full"][0]
    objects = ranks["objects_only"][0]
    for direction, mask in [("full_wins", (full == 1) & (objects > 1)), ("objects_win", (objects == 1) & (full > 1))]:
        for index in np.flatnonzero(mask)[:5]:
            qid = data.query_ids[index]
            graph = graph_of[int(true_images[index])]
            cases.append(
                {
                    "group": direction,
                    "query_id": qid,
                    "text": query_of[qid]["text"],
                    "image_id": int(true_images[index]),
                    "seed": config.training.seeds[0],
                    "full_rank": int(full[index]),
                    "objects_rank": int(objects[index]),
                    "concepts": [concepts[c] for c in graph["concept_ids"]],
                    "triplets": [[concepts[s], predicates[p], concepts[o]] for s, p, o in graph["triplets"]],
                    "selection": "first five query IDs in each direction; descriptive, no annotation of query type",
                }
            )

    # Persist comparisons, case studies and input provenance.
    write_json(
        Path(output),
        {
            "official": official,
            "paired": comparisons,
            "posthoc_fixed_k500_existing_alpha": fixed_k,
            "diagnostic_warning": "fixed k is not a replacement benchmark or selected hyperparameter",
            "case_studies": cases,
            "inputs_sha256": {str(p): sha256_of(p) for p in inputs},
            "protocol": {"config": config_path, "split": "test", "training_seeds": list(config.training.seeds)},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval/ablations/full.yaml")
    parser.add_argument("--output", default="runs/analysis/ablation.json")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
