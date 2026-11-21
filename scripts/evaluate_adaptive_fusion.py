#!/usr/bin/env python3
"""Query-adaptive fusion: one validation-selected (alpha, top_k) per bucket of queries.

Run from the repository root after train_graphsage and run_benchmark for the configured run:
    python -m scripts.evaluate_adaptive_fusion
    python -m scripts.evaluate_adaptive_fusion --config configs/evaluation/adaptive_fusion.yaml

For every training seed:
- the baseline is the official fixed-alpha reranker, selected on all validation queries; it
  must reproduce the benchmark's selection;
- for every (feature, number of buckets), buckets are cut on the validation feature and
  (alpha, top_k) is selected per bucket on validation;
- validation and test are then scored once with those settings.

The random feature is a control with the same number of selected parameters. Nothing is
retrained and nothing is selected on test.

Writes the JSON named by `output` in the config: the selected settings, metrics overall and
per bucket, mean ± std over seeds, and a paired, image-clustered bootstrap of each bucketing
against the fixed-alpha baseline on test.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import META_FILE, load_table, table_paths, verify_inputs_unchanged
from src.evaluation.ablation_analysis import paired_analysis
from src.evaluation.benchmark import aggregate_runs, build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.models.checkpoint import GRAPH_VECTORS, SETTINGS_FILE
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.adaptive_fusion import AdaptiveFusionConfig, adaptive_scores, fit_adaptive_fusion
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import SweepPoint, rerank_scores, sweep_alpha_and_k

log = logging.getLogger("evaluate_adaptive_fusion")

DEFAULT_CONFIG = "configs/evaluation/adaptive_fusion.yaml"
EVAL_SPLITS = (Split.VAL, Split.TEST)
FIXED = "fixed"


def _check_official(selected: SweepPoint, benchmark: dict[str, Any], seed: int, label: str) -> None:
    """The baseline must be exactly what run_benchmark selected for this seed."""
    official = benchmark["seeds"][str(seed)]["selected_on_validation"]
    if (selected.alpha, selected.top_k) != (official["alpha"], official["top_k"]):
        raise DataError(
            f"seed {seed}: the official sweep selected alpha={selected.alpha}, top_k={selected.top_k}, but {label} "
            f"selected alpha={official['alpha']}, top_k={official['top_k']}; the grids or inputs differ"
        )


def _bucket_rows(
    ranks: np.ndarray, fixed_ranks: np.ndarray, buckets: np.ndarray, n_buckets: int
) -> list[dict[str, Any]]:
    """Size and Recall@1 of each bucket, in the order of `fit.buckets`, next to the fixed baseline's."""
    rows = []
    for bucket in range(n_buckets):
        mask = buckets == bucket
        rows.append(
            {
                "queries": int(mask.sum()),
                "recall@1": float((ranks[mask] == 1).mean()) if mask.any() else None,
                "fixed_recall@1": float((fixed_ranks[mask] == 1).mean()) if mask.any() else None,
            }
        )
    return rows


def _log_seed(seed: int, key: str, result: dict[str, Any]) -> None:
    log.info(
        "seed %d %-16s val R@1 %.4f  test R@1 %.4f",
        seed,
        key,
        result[Split.VAL.value]["metrics"]["recall@1"],
        result[Split.TEST.value]["metrics"]["recall@1"],
    )


def run(config: AdaptiveFusionConfig, config_path: str) -> None:
    """Fit the baseline and every bucketing on validation per seed, score validation and test, write the report."""
    retrieval = RetrievalConfig.from_yaml(config.retrieval_config)
    data_config = DataConfig.from_yaml(retrieval.data_config)
    processed = data_config.paths.processed_dir
    embeddings = retrieval.model_embeddings_dir()
    verify_inputs_unchanged(embeddings)
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")
    image_table, query_table = load_table(embeddings, "images"), load_table(embeddings, "queries")
    data = {split: build_split_data(split, images, queries, image_table, query_table) for split in EVAL_SPLITS}
    clip = {split: clip_scores(split_data) for split, split_data in data.items()}

    # The benchmark of this run supplies the official selection that the baseline must reproduce.
    name = retrieval.graphsage.run_name()
    benchmark_path = retrieval.runs_dir / "benchmark" / f"graphsage-{name}.json"
    benchmark = read_json(benchmark_path)
    validate_artifact_inputs(benchmark, str(benchmark_path))
    grid = retrieval.benchmark
    inputs = [
        Path(config_path),
        Path(config.retrieval_config),
        embeddings / META_FILE,
        data_config.paths.split_file,
        benchmark_path,
    ]

    plans = {f"{feature.value}-b{n}": (feature, n) for feature in config.features for n in config.n_buckets}
    entries: dict[str, dict[str, Any]] = {FIXED: {"seeds": {}}}
    entries |= {key: {"feature": feature.value, "n_buckets": n, "seeds": {}} for key, (feature, n) in plans.items()}
    test_ranks: dict[str, list[np.ndarray]] = {key: [] for key in entries}
    for seed in retrieval.training.seeds:
        run_dir = retrieval.runs_dir / "graphsage" / name / f"seed{seed}"
        validate_run(run_dir, retrieval, seed)
        inputs += [run_dir / SETTINGS_FILE, *table_paths(run_dir, GRAPH_VECTORS)]
        table = load_table(run_dir, GRAPH_VECTORS)
        graph = {split: graph_vector_scores(split_data, table) for split, split_data in data.items()}

        # The baseline is the official reranker: one (alpha, top_k) for every query.
        selected, _ = sweep_alpha_and_k(
            clip[Split.VAL], graph[Split.VAL], data[Split.VAL], grid.alpha_grid, grid.top_k_grid, grid.recall_ks
        )
        _check_official(selected, benchmark, seed, str(benchmark_path))
        fixed: dict[str, Any] = {"fit": {"alpha": selected.alpha, "top_k": selected.top_k}}
        fixed_ranks: dict[Split, np.ndarray] = {}
        for split in EVAL_SPLITS:
            fused = rerank_scores(clip[split], graph[split], selected.alpha, selected.top_k)
            fixed_ranks[split] = ranks_of_relevant(fused, data[split].relevant_columns)
            fixed[split.value] = {"metrics": retrieval_metrics(fixed_ranks[split], grid.recall_ks)}
        entries[FIXED]["seeds"][str(seed)] = fixed
        test_ranks[FIXED].append(fixed_ranks[Split.TEST])
        _log_seed(seed, FIXED, fixed)

        for key, (feature, n_buckets) in plans.items():
            fusion = fit_adaptive_fusion(
                clip[Split.VAL],
                graph[Split.VAL],
                data[Split.VAL],
                feature,
                n_buckets,
                grid.alpha_grid,
                grid.top_k_grid,
                min_bucket_queries=config.min_bucket_queries,
                feature_seed=config.feature_seed,
                ks=grid.recall_ks,
            )
            result: dict[str, Any] = {"fit": fusion.to_dict()}
            for split in EVAL_SPLITS:
                fused, buckets = adaptive_scores(fusion, clip[split], graph[split], data[split].query_ids)
                ranks = ranks_of_relevant(fused, data[split].relevant_columns)
                result[split.value] = {
                    "metrics": retrieval_metrics(ranks, grid.recall_ks),
                    "buckets": _bucket_rows(ranks, fixed_ranks[split], buckets, len(fusion.settings)),
                }
                if split is Split.TEST:
                    test_ranks[key].append(ranks)
            entries[key]["seeds"][str(seed)] = result
            _log_seed(seed, key, result)

    # Paired comparison with the fixed baseline: same queries, same seeds, captions clustered by image.
    test = data[Split.TEST]
    true_images = np.asarray(test.image_ids)[test.relevant_columns]
    fixed_test_ranks = np.stack(test_ranks[FIXED])
    for key, entry in entries.items():
        entry["summary"] = {
            split.value: aggregate_runs([result[split.value]["metrics"] for result in entry["seeds"].values()])
            for split in EVAL_SPLITS
        }
        if key != FIXED:
            entry["paired_vs_fixed_test"] = paired_analysis(
                np.stack(test_ranks[key]),
                fixed_test_ranks,
                true_images,
                repeats=config.bootstrap_repeats,
                seed=config.bootstrap_seed,
            )
        summary = entry["summary"][Split.TEST.value]["recall@1"]
        log.info("%-16s test R@1 %.4f ± %.4f", key, summary["mean"], summary["std"])

    write_json(
        Path(config.output),
        {
            "run_name": name,
            "clip_model": retrieval.clip.model_name,
            "protocol": {
                "config": config_path,
                "retrieval_config": config.retrieval_config,
                "training_seeds": list(retrieval.training.seeds),
                "fixed": "the official reranker: one (alpha, top_k) for all queries, equal to the benchmark's",
                "selection": "per bucket on validation with the official alpha/top_k sweep and tie-break",
                "buckets": "edges at quantiles of the validation feature; test queries use the validation edges",
                "control": "random feature: same bucket sizes and parameter count, no information",
                "paired_vs_fixed_test": "first = this bucketing, second = fixed; delta_r1_pp = bucketing − fixed",
                "min_bucket_queries": config.min_bucket_queries,
                "feature_seed": config.feature_seed,
            },
            "results": entries,
            "inputs_sha256": {str(path): sha256_of(path) for path in inputs},
        },
    )
    log.info("wrote %s", config.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(AdaptiveFusionConfig.from_yaml(args.config), args.config)
    except AppError:
        log.exception("adaptive fusion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
