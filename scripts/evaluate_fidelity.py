#!/usr/bin/env python3
"""Evaluate true-image relation fidelity using an existing VG–COCO checkpoint.

No training or parameter search. The sample, budget, alpha and checkpoint are recorded;
this measures score sensitivity, not retrieval-rank changes or semantic validity.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch

from src.core.config_parsing import read_mapping
from src.core.exceptions import AppError, ConfigError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import META_FILE, load_table, verify_inputs_unchanged
from src.evaluation.benchmark import build_split_data
from src.evaluation.fidelity import evaluate_fidelity
from src.models.checkpoint import BEST_FILE, SETTINGS_FILE, RunDirectory
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import clip_candidate_mask

log = logging.getLogger("evaluate_fidelity")
DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"


def _verify_hashes(hashes: dict[str, str]) -> None:
    changed = [name for name, digest in hashes.items() if not Path(name).is_file() or sha256_of(Path(name)) != digest]
    if changed:
        raise DataError(f"evaluation inputs changed: {changed}")


def run(config: RetrievalConfig, name: str, seed: int, split: Split, sample: int, budget: int) -> dict[str, Any]:
    """Load unchanged artifacts and evaluate a seed-fixed sample of the split's queries."""
    if split not in (Split.VAL, Split.TEST) or sample < 1 or budget < 1 or seed < 0:
        raise ConfigError("need val/test, positive sample/budget and a nonnegative seed")
    directory = RunDirectory(config.runs_dir / "graphsage" / name / f"seed{seed}")
    benchmark_path = config.runs_dir / "benchmark" / f"graphsage-{name}.json"
    if not benchmark_path.is_file() or not (directory.path / SETTINGS_FILE).is_file():
        raise DataError("run train_graphsage and run_benchmark for this run before fidelity evaluation")
    benchmark = read_json(benchmark_path)
    settings = read_mapping(directory.path / SETTINGS_FILE)
    if settings["clip_model"] != config.clip.model_name or settings["seed"] != seed:
        raise DataError("training settings differ from the requested CLIP model or seed")
    _verify_hashes(settings["inputs_sha256"])
    _verify_hashes(benchmark["inputs_sha256"])
    selected = benchmark.get("seeds", {}).get(str(seed), {}).get("selected_on_validation")
    if not selected:
        raise DataError(f"benchmark has no validation-selected parameters for seed {seed}")
    alpha = float(selected["alpha"])
    embeddings = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings)
    images, queries = load_table(embeddings, "images"), load_table(embeddings, "queries")
    concepts, predicates = load_table(embeddings, "concepts"), load_table(embeddings, "predicates")
    if concepts.ids != tuple(range(len(concepts.ids))) or predicates.ids != tuple(range(len(predicates.ids))):
        raise DataError("concept and predicate embedding rows must match their integer ids")
    raw_model = settings["graphsage"]
    model_config = GraphSageConfig(**{**raw_model, "image_node": ImageNodeFeature(raw_model["image_node"])})
    model = GraphImageEncoder(
        model_config,
        torch.from_numpy(concepts.vectors),
        torch.from_numpy(predicates.vectors),
        torch.from_numpy(images.vectors),
    )
    model.load_state_dict(directory.load_best()["model"])
    data_config = DataConfig.from_yaml(config.data_config)
    processed = data_config.paths.processed_dir
    graph_path, query_path = processed / "graph/image_graphs.json", processed / "queries.json"
    for path in (embeddings / META_FILE, graph_path, data_config.paths.split_file):
        if settings["inputs_sha256"].get(str(path)) != sha256_of(path):
            raise DataError(f"active input differs from the trained checkpoint: {path}")
    for path in (embeddings / META_FILE, data_config.paths.split_file):
        if benchmark["inputs_sha256"].get(str(path)) != sha256_of(path):
            raise DataError(f"active input differs from the validation benchmark: {path}")
    graphs = {int(g["vg_image_id"]): g for g in read_json(graph_path)}
    query_records = read_json(query_path)
    split_data = build_split_data(split, read_json(processed / "images.json"), query_records, images, queries)
    pool = sorted((q for q in query_records if q["split"] == split.value), key=lambda q: q["query_id"])
    columns = {image_id: column for column, image_id in enumerate(split_data.image_ids)}
    if sample > len(pool):
        raise ConfigError(f"sample {sample} exceeds the {len(pool)} queries of split {split.value}")
    query_rows, image_rows = queries.row_of(), images.row_of()
    records = []
    for query in random.Random(seed).sample(pool, sample):
        if len(query["relevant_ids"]) != 1:
            raise DataError("fidelity requires exactly one relevant image per query")
        image_id = int(query["relevant_ids"][0])
        if image_id not in graphs or image_id not in image_rows or query["query_id"] not in query_rows:
            raise DataError(f"missing graph or embedding for query {query['query_id']}")
        query_vector = queries.vectors[query_rows[query["query_id"]]]
        clip = (split_data.image_vectors @ query_vector)[None, :]
        candidates = clip_candidate_mask(clip, min(int(selected["top_k"]), len(split_data.image_ids)))
        result = evaluate_fidelity(
            model,
            graphs[image_id],
            image_rows[image_id],
            torch.from_numpy(query_vector),
            budget=budget,
            seed=seed,
            alpha=alpha,
            is_candidate=bool(candidates[0, columns[image_id]]),
        )
        records.append({"query_id": query["query_id"], "image_id": image_id, **result.to_dict()})
    inputs = [
        directory.path / BEST_FILE,
        directory.path / SETTINGS_FILE,
        benchmark_path,
        embeddings / META_FILE,
        graph_path,
        query_path,
        processed / "images.json",
    ]
    return {
        "run_name": name,
        "seed": seed,
        "split": split.value,
        "alpha": alpha,
        "top_k_selected_on_validation": selected["top_k"],
        "sample": sample,
        "budget": budget,
        "protocol": "fixed checkpoint; true-image score; positive leave-one-out support; disjoint equal-budget control",
        "limitations": [
            "no rank-change measurement",
            "no semantic validity claim",
            "budget capped at half the relations",
        ],
        "inputs_sha256": {str(path): sha256_of(path) for path in inputs},
        "queries": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="training run; defaults to the config's architecture")
    parser.add_argument("--seed", type=int, help="trained seed; defaults to the first configured seed")
    parser.add_argument("--split", choices=[Split.VAL.value, Split.TEST.value], default=Split.TEST.value)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, required=True, help="new result file; existing files are never overwritten"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.output.exists():
            raise DataError(f"output already exists: {args.output}")
        config = RetrievalConfig.from_yaml(args.config)
        result = run(
            config,
            args.run or config.graphsage.run_name(),
            args.seed if args.seed is not None else config.training.seeds[0],
            Split(args.split),
            args.sample,
            args.budget,
        )
        write_json(args.output, result)
    except AppError:
        log.exception("fidelity evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
