#!/usr/bin/env python3
"""Frozen full-model joint-deletion diagnostic; no retraining or parameter selection.

python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity.yaml
python -m scripts.evaluate_fidelity --config configs/evaluation/fidelity_gnn_explainer.yaml

`selector` in the protocol picks how relations are ranked before they are removed:
single_deletion (default) or gnn_explainer. Queries, budgets and random groups are the same
for both, so the two outputs compare row by row.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.config_parsing import read_mapping, require_field
from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.benchmark import build_split_data, clip_scores, graph_vector_scores
from src.evaluation.fidelity import MIN_RELATIONS, joint_relation_fidelity, ranking_effects
from src.models.ablations import Ablation
from src.models.checkpoint import RunDirectory
from src.models.gnn_explainer import GnnExplainerConfig, gnn_explainer_relation_scores
from src.models.graphsage import GraphImageEncoder
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger(__name__)

class Selector(StrEnum):
    """How the relations of an explanation are ranked before the joint deletion."""

    SINGLE_DELETION = "single_deletion"
    GNN_EXPLAINER = "gnn_explainer"


LIMITATIONS = {
    Selector.SINGLE_DELETION: (
        "Selected by single-deletion graph score; joint/rank tests are still selection-dependent diagnostics"
    ),
    Selector.GNN_EXPLAINER: (
        "Selected by a learned GNNExplainer edge mask, independent of deletion; the mask depends on its random start"
    ),
}


def run(path: str) -> None:
    """Read frozen protocol, evaluate all configured training seeds, retain per-query outcomes."""
    # Read the frozen protocol before selecting any queries.
    protocol = read_mapping(path)
    config = RetrievalConfig.from_yaml(require_field(protocol, "retrieval_config", str))
    if config.graphsage.ablation is not Ablation.FULL:
        raise DataError("this protocol evaluates full-model explanations only")
    split = Split(require_field(protocol, "split", str))
    if split is Split.TRAIN:
        raise DataError("fidelity must use validation or test")
    sample = require_field(protocol, "sample_queries", int)
    repeats = require_field(protocol, "random_repeats", int)
    batch_size = require_field(protocol, "batch_size", int)
    sample_seed = require_field(protocol, "sample_seed", int)
    random_seed = require_field(protocol, "random_seed", int)
    selector = Selector(protocol.get("selector", Selector.SINGLE_DELETION.value))
    explainer_config = None
    if selector is Selector.GNN_EXPLAINER:
        settings = require_field(protocol, "gnn_explainer", dict)
        explainer_config = GnnExplainerConfig(
            require_field(settings, "epochs", int),
            float(require_field(settings, "learning_rate", float)),
            config.training.temperature,
        )
    fractions = protocol.get("fractions")
    if (
        not isinstance(fractions, list)
        or not fractions
        or any(isinstance(f, bool) or not isinstance(f, int | float) or not 0 < f < 1 for f in fractions)
        or len(set(fractions)) != len(fractions)
        or min(sample, repeats, batch_size) < 1
    ):
        raise DataError("invalid fidelity fractions, sample size, repeats or batch size")

    # Load the same split and embedding tables used by retrieval.
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    embeddings = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings)
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")
    image_table = load_table(embeddings, "images")
    query_table = load_table(embeddings, "queries")
    data = build_split_data(
        split,
        images,
        queries,
        image_table,
        query_table,
    )

    # Fix one reproducible query sample for every model seed.
    if sample > len(data.query_ids):
        raise DataError("sample_queries exceeds queries in the split")
    rows = sorted(random.Random(sample_seed).sample(range(len(data.query_ids)), sample))
    query_of = {q["query_id"]: q for q in queries}
    graphs = {int(g["vg_image_id"]): g for g in read_json(processed / "graph/image_graphs.json")}
    concepts = load_table(embeddings, "concepts")
    predicates = load_table(embeddings, "predicates")
    if list(concepts.ids) != list(range(len(concepts.ids))) or list(predicates.ids) != list(range(len(predicates.ids))):
        raise DataError("concept and predicate tables must have dense ordered IDs")

    # Reuse the benchmark selection; reject stale inputs.
    clip = clip_scores(data)
    name = config.graphsage.run_name()
    benchmark_path = config.runs_dir / "benchmark" / f"graphsage-{name}.json"
    benchmark = read_json(benchmark_path)
    validate_artifact_inputs(benchmark, str(benchmark_path))
    records: dict[str, Any] = {}
    inputs = [Path(path), Path(protocol["retrieval_config"]), benchmark_path, embeddings / "meta.json"]
    for seed in config.training.seeds:
        # Restore the trained encoder and its exported graph scores.
        directory = config.runs_dir / "graphsage" / name / f"seed{seed}"
        validate_run(
            directory,
            config,
            seed,
        )
        model = GraphImageEncoder(
            config.graphsage,
            torch.from_numpy(concepts.vectors),
            torch.from_numpy(predicates.vectors),
            torch.from_numpy(image_table.vectors),
        )
        model.load_state_dict(RunDirectory(directory).load_best()["model"])
        model.eval()
        graph = graph_vector_scores(data, load_table(directory, "graph_vectors"))
        choice = benchmark["seeds"][str(seed)]["selected_on_validation"]
        alpha, top_k = choice["alpha"], choice["top_k"]
        final = rerank_scores(
            clip,
            graph,
            alpha,
            top_k,
        )
        seed_records = []
        for done, row in enumerate(rows, 1):
            # Perturb the original prediction while all rival scores stay fixed.
            target = int(np.argmax(final[row]))
            image_id = data.image_ids[target]
            query_vector = torch.from_numpy(data.query_vectors[row])
            image_row = image_table.row_of()[image_id]
            importance = None
            if explainer_config is not None and len(graphs[image_id]["triplets"]) >= MIN_RELATIONS:
                importance = gnn_explainer_relation_scores(
                    model, graphs[image_id], image_row, query_vector, explainer_config, random_seed + row
                ).tolist()
            result = joint_relation_fidelity(
                model,
                graphs[image_id],
                image_row,
                query_vector,
                fractions,
                repeats,
                random_seed + row,
                batch_size,
                importance,
            )
            result.update(
                query_id=data.query_ids[row],
                text=query_of[data.query_ids[row]]["text"],
                image_id=image_id,
                predicted_image_is_relevant=target == int(data.relevant_columns[row]),
            )

            # Check score agreement before measuring rank changes.
            if result["status"] == "evaluated":
                if abs(result["baseline_graph_score"] - float(graph[row, target])) > 1e-5:
                    raise DataError("checkpoint score differs from exported graph vectors")
                for budget in result["budgets"]:
                    budget.update(
                        ranking_effects(
                            final[row],
                            target,
                            float(clip[row, target]),
                            alpha,
                            budget,
                        )
                    )
            seed_records.append(result)
            if done % 25 == 0:
                log.info("seed %s: %d/%d queries", seed, done, sample)

        # Aggregate evaluated queries and retain skipped cases in the trace.
        summary = []
        for fraction in fractions:
            budgets = [
                b
                for r in seed_records
                if r["status"] == "evaluated"
                for b in r["budgets"]
                if b["fraction_requested"] == fraction
            ]
            keys = (
                "selected_drop",
                "random_drop_mean",
                "excess_drop",
                "positive_drop",
                "beats_random_mean",
                "selected_rank_increase",
                "random_rank_increase_mean",
                "selected_top1_lost",
                "random_top1_lost_rate",
                "overlap_with_single_deletion",
            )
            summary.append(
                {
                    "fraction": fraction,
                    "evaluated": len(budgets),
                    **{k: float(np.mean([b[k] for b in budgets])) if budgets else None for k in keys},
                }
            )
        records[str(seed)] = {"selected_on_validation": choice, "summary": summary, "queries": seed_records}
        inputs += [directory / "config.yaml", directory / "best.pt", directory / "graph_vectors.npy"]

    # Save per-query evidence and hashes so this run can be audited.
    output = {
        "protocol": protocol,
        "sampled_query_ids": [data.query_ids[i] for i in rows],
        "seeds": records,
        "target": "original_predicted_top1_image; all rival scores and CLIP candidates frozen",
        "selector": selector.value,
        "limitation": LIMITATIONS[selector],
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs},
    }
    write_json(
        Path(require_field(protocol, "output", str)),
        output,
    )
    log.info("wrote %s", protocol["output"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/fidelity.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args.config)
    except (AppError, ValueError):
        log.exception("fidelity failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
