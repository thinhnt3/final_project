#!/usr/bin/env python3
"""Measure relation sparsity/retention, export validity cases, or summarize supplied labels.

python -m scripts.evaluate_explanation_quality
python -m scripts.evaluate_explanation_quality --labels review-v1.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from src.core.config_parsing import read_mapping, require_field
from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.download import image_path
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import load_table, verify_inputs_unchanged
from src.evaluation.benchmark import build_split_data, clip_scores, graph_vector_scores
from src.evaluation.explanation_quality import (
    keep_only_scores,
    relation_sparsity,
    select_review_queries,
    summarize_quality,
    summarize_validity,
)
from src.evaluation.fidelity import ranking_effects
from src.models.ablations import Ablation
from src.models.checkpoint import RunDirectory
from src.models.graphsage import GraphImageEncoder
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger(__name__)
DEFAULT_CONFIG = "configs/evaluation/explanation_quality.yaml"


def run(path: str) -> None:
    """Keep the frozen explanation subsets and score them without retraining."""
    protocol = read_mapping(path)
    source_path = Path(require_field(protocol, "fidelity_input", str))
    source = read_json(source_path)
    validate_artifact_inputs(source, str(source_path))
    config = RetrievalConfig.from_yaml(source["protocol"]["retrieval_config"])
    if config.graphsage.ablation is not Ablation.FULL:
        raise DataError("explanation quality expects the full-model fidelity trace")
    batch_size = require_field(protocol, "batch_size", int)
    if batch_size < 1:
        raise DataError("batch_size must be positive")

    # Restore the exact gallery, query vectors and graph inputs used by fidelity.
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    embeddings = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings)
    image_table = load_table(embeddings, "images")
    concepts = load_table(embeddings, "concepts")
    predicates = load_table(embeddings, "predicates")
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")
    graphs = {int(g["vg_image_id"]): g for g in read_json(processed / "graph/image_graphs.json")}
    data = build_split_data(
        Split(source["protocol"]["split"]),
        images,
        queries,
        image_table,
        load_table(embeddings, "queries"),
    )
    query_rows = {query_id: row for row, query_id in enumerate(data.query_ids)}
    image_rows = image_table.row_of()
    clip = clip_scores(data)
    inputs = [
        Path(path),
        source_path,
        Path("scripts/evaluate_explanation_quality.py"),
        Path("src/evaluation/explanation_quality.py"),
        processed / "images.json",
        processed / "queries.json",
        processed / "graph/image_graphs.json",
        processed / "graph/concepts.json",
        processed / "graph/predicates.json",
    ]
    records = {}
    for seed in config.training.seeds:
        directory = config.runs_dir / "graphsage" / config.graphsage.run_name() / f"seed{seed}"
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
        graph_scores = graph_vector_scores(data, load_table(directory, "graph_vectors"))
        source_seed = source["seeds"][str(seed)]
        choice = source_seed["selected_on_validation"]
        alpha, top_k = float(choice["alpha"]), int(choice["top_k"])
        final = rerank_scores(
            clip,
            graph_scores,
            alpha,
            top_k,
        )
        seed_queries = []
        for done, original in enumerate(source_seed["queries"], 1):
            record = {key: value for key, value in original.items() if key != "budgets"}
            if original["status"] != "evaluated":
                seed_queries.append(record)
                continue
            row = query_rows[original["query_id"]]
            target = int(np.argmax(final[row]))
            image_id = data.image_ids[target]
            if (
                image_id != original["image_id"]
                or len(graphs[image_id]["triplets"]) != original["relations"]
                or abs(float(graph_scores[row, target]) - original["baseline_graph_score"]) > 1e-5
            ):
                raise DataError("fidelity target/graph score no longer matches the frozen model")
            budgets = []
            for budget in original["budgets"]:
                keep = keep_only_scores(
                    model,
                    graphs[image_id],
                    image_rows[image_id],
                    torch.from_numpy(data.query_vectors[row]),
                    budget,
                    batch_size,
                )
                effect = ranking_effects(
                    final[row],
                    target,
                    float(clip[row, target]),
                    alpha,
                    keep,
                )
                # Preserve signed differences; retention can raise or lower the score.
                budgets.append(
                    {
                        "fraction_requested": budget["fraction_requested"],
                        "selected_indices": budget["selected_indices"],
                        **relation_sparsity(original["relations"], budget["selected_indices"]),
                        "deletion_graph_drop": budget["selected_drop"],
                        "deletion_top1_lost": budget["selected_top1_lost"],
                        "keep_graph_score": keep["selected_score"],
                        "keep_graph_drop": original["baseline_graph_score"] - keep["selected_score"],
                        "keep_random_graph_scores": keep["random_scores"],
                        "keep_random_graph_drop": original["baseline_graph_score"]
                        - float(np.mean(keep["random_scores"])),
                        "baseline_rank": effect["baseline_rank"],
                        "keep_top1_preserved": effect["baseline_rank"] == 1 and effect["selected_rank"] == 1,
                        "keep_random_top1_preserved": (
                            1 - effect["random_top1_lost_rate"] if effect["baseline_rank"] == 1 else 0.0
                        ),
                        "keep_rank_increase": effect["selected_rank_increase"],
                    }
                )
            record["budgets"] = budgets
            seed_queries.append(record)
            if done % 25 == 0:
                log.info("seed %s: %d/%d retained subsets", seed, done, len(source_seed["queries"]))
        records[str(seed)] = {
            "selected_on_validation": choice,
            "summary": summarize_quality(seed_queries),
            "queries": seed_queries,
        }

    # Export a balanced, seeded review sample, with no automatic semantic truth labels.
    review_seed = require_field(protocol, "review_model_seed", int)
    review_fraction = require_field(protocol, "review_fraction", (int, float))
    if str(review_seed) not in source["seeds"] or review_fraction not in source["protocol"]["fractions"]:
        raise DataError("review seed/fraction must exist in the fidelity trace")
    concept_text = {c["concept_id"]: c["text"] for c in read_json(processed / "graph/concepts.json")}
    predicate_text = {p["predicate_id"]: p["text"] for p in read_json(processed / "graph/predicates.json")}
    sample = select_review_queries(
        source["seeds"][str(review_seed)]["queries"],
        require_field(protocol, "review_per_stratum", int),
        require_field(protocol, "review_sample_seed", int),
    )
    cases = []
    for query in sample:
        image_id = query["image_id"]
        photo = image_path(processed / "images", image_id)
        if not photo.is_file():
            raise DataError(f"validity image missing: {photo}")
        budget = next(b for b in query["budgets"] if b["fraction_requested"] == review_fraction)
        relations = []
        for index, drop in zip(budget["selected_indices"], budget["single_drops"], strict=True):
            subject, predicate, obj = graphs[image_id]["triplets"][index]
            relations.append(
                {
                    "index": index,
                    "triplet": [subject, predicate, obj],
                    "subject": concept_text[subject],
                    "predicate": predicate_text[predicate],
                    "object": concept_text[obj],
                    "single_deletion_drop": drop,
                }
            )
        cases.append(
            {
                "case_id": f"{review_seed}:{query['query_id']}",
                "seed": review_seed,
                "query_id": query["query_id"],
                "text": query["text"],
                "image_id": image_id,
                "image_path": str(photo),
                "image_sha256": sha256_of(photo),
                "predicted_image_is_relevant": query["predicted_image_is_relevant"],
                "available_relations": query["relations"],
                "fraction_requested": review_fraction,
                "relations": relations,
            }
        )
        inputs.append(photo)
    review_path = Path(require_field(protocol, "review_output", str))
    write_json(
        review_path,
        {
            "protocol": protocol,
            "sampling": "balanced relevant/non-relevant top1 predictions; not representative prevalence",
            "rubric": {
                "image_support": "yes/no/uncertain: subject-predicate-object is visually supported in the image",
                "query_relevance": (
                    "yes/no/uncertain: the relation directly supports a requested entity/action/constraint"
                ),
                "uncertain": "Use when the image, identity, geometry or wording is insufficient to decide.",
                "joint_validity": "yes only when both dimensions are yes; no if either is no; otherwise uncertain",
            },
            "cases": cases,
        },
    )
    inputs.append(review_path)
    write_json(
        Path(require_field(protocol, "output", str)),
        {
            "protocol": protocol,
            "source_fidelity": str(source_path),
            "seeds": records,
            "validity": {"status": "pending_labels", "cases": len(cases)},
            "limitations": [
                "Relation sparsity only; objects/image-object links retained, not a minimal sufficient graph.",
                "Keep-only rank preservation can be dominated by frozen CLIP; inspect graph score and random control.",
                "Selected by single-deletion effects; retention is still selection-dependent.",
                "Additional posthoc analysis on already observed test results; no tuning or retraining.",
            ],
            "inputs_sha256": {
                **source["inputs_sha256"],
                **{str(p): sha256_of(p) for p in inputs},
            },
        },
    )
    log.info("wrote %s and %s", protocol["output"], review_path)


def score_review(path: str, labels_path: str) -> None:
    """Score a completed AI/human review without repeating checkpoint inference."""
    protocol = read_mapping(path)
    output_path = Path(require_field(protocol, "output", str))
    quality = read_json(output_path)
    validate_artifact_inputs(quality, str(output_path))
    if quality["protocol"] != protocol:
        raise DataError("quality artifact was generated with a different protocol")
    review_path = Path(require_field(protocol, "review_output", str))
    labels = read_json(labels_path)
    if labels.get("review_manifest_sha256") != sha256_of(review_path):
        raise DataError("validity labels do not match the exported review manifest")
    summary = summarize_validity(
        read_json(review_path)["cases"],
        labels,
    )
    write_json(
        output_path.with_name("validity-summary.json"),
        {
            **summary,
            "inputs_sha256": {
                str(output_path): sha256_of(output_path),
                str(review_path): sha256_of(review_path),
                labels_path: sha256_of(Path(labels_path)),
            },
        },
    )
    log.info("wrote %s", output_path.with_name("validity-summary.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--labels", help="summarize a complete v1 review; reuse the existing inference artifact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.labels:
            score_review(args.config, args.labels)
        else:
            run(args.config)
    except (AppError, ValueError):
        log.exception("explanation quality evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
