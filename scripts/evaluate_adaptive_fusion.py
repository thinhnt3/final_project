#!/usr/bin/env python3
"""Query-adaptive fusion: one validation-selected (alpha, top_k) per bucket of queries.

Run from the repository root after train_graphsage and run_benchmark for the configured run:
    python -m scripts.evaluate_adaptive_fusion
    python -m scripts.evaluate_adaptive_fusion --config configs/evaluation/adaptive_fusion.yaml

For every training seed and every (feature, number of buckets):
- buckets are cut on the validation feature;
- (alpha, top_k) is selected per bucket on validation;
- validation and test are then scored once with those settings.

One bucket is the official fixed-alpha reranker and must reproduce the benchmark's selection.
The random feature is a control with the same number of selected parameters. Nothing is
retrained and nothing is selected on test.

Writes the JSON named by `output` in the config: per-bucket settings and metrics, mean ± std
over seeds, and a paired, image-clustered bootstrap of each bucketing against the
fixed-alpha baseline on test.
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
from src.embeddings.store import META_FILE, load_table, verify_inputs_unchanged
from src.evaluation.ablation_analysis import paired_analysis
from src.evaluation.benchmark import aggregate_runs, build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.models.checkpoint import GRAPH_VECTORS, SETTINGS_FILE
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.adaptive_fusion import (
    AdaptiveFusion,
    AdaptiveFusionConfig,
    BucketFeature,
    adaptive_scores,
    fit_adaptive_fusion,
)
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("evaluate_adaptive_fusion")

DEFAULT_CONFIG = "configs/evaluation/adaptive_fusion.yaml"
EVAL_SPLITS = (Split.VAL, Split.TEST)
FIXED = "fixed"
_DECIMALS = 4
_ALPHA_TOLERANCE = 1e-9


def _key(feature: BucketFeature, n_buckets: int) -> str:
    return FIXED if n_buckets == 1 else f"{feature.value}-b{n_buckets}"


def _check_official(fusion: AdaptiveFusion, benchmark: dict[str, Any], seed: int, label: str) -> None:
    """With one bucket the selection must equal what run_benchmark selected for this seed."""
    official = benchmark["seeds"][str(seed)]["selected_on_validation"]
    (setting,) = fusion.settings
    if abs(setting.alpha - float(official["alpha"])) > _ALPHA_TOLERANCE or setting.top_k != int(official["top_k"]):
        raise DataError(
            f"seed {seed}: one bucket selected alpha={setting.alpha}, top_k={setting.top_k}, but {label} "
            f"selected alpha={official['alpha']}, top_k={official['top_k']}; the grids or inputs differ"
        )


def _bucket_rows(
    ranks: np.ndarray, fixed_ranks: np.ndarray, buckets: np.ndarray, fusion: AdaptiveFusion
) -> list[dict[str, Any]]:
    """Per bucket: size, setting, and Recall@1 of this bucketing and of the fixed baseline on the same queries."""
    rows = []
    for bucket, setting in enumerate(fusion.settings):
        mask = buckets == bucket
        count = int(mask.sum())
        rows.append(
            {
                "bucket": bucket,
                "queries": count,
                "alpha": setting.alpha,
                "top_k": setting.top_k,
                "recall@1": round(float((ranks[mask] == 1).mean()), _DECIMALS) if count else None,
                "fixed_recall@1": round(float((fixed_ranks[mask] == 1).mean()), _DECIMALS) if count else None,
            }
        )
    return rows


def _summary(entry: dict[str, Any], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Mean and sample std over seeds of every metric, per split."""
    return {
        split.value: {
            metric: {stat: round(value, _DECIMALS) for stat, value in stats.items()}
            for metric, stats in aggregate_runs([entry["seeds"][str(s)][split.value]["metrics"] for s in seeds]).items()
        }
        for split in EVAL_SPLITS
    }


def run(config: AdaptiveFusionConfig, config_path: str) -> None:
    """Fit every bucketing on validation per seed, score validation and test, and write the report."""
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

    # The benchmark of this run supplies the official selection that one bucket must reproduce.
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

    # The fixed baseline first: every other bucketing is compared with it on the same queries.
    plans = [(BucketFeature.CLIP_MARGIN, 1)] + [(f, b) for f in config.features for b in config.n_buckets if b > 1]
    entries: dict[str, dict[str, Any]] = {}
    test_ranks: dict[str, list[np.ndarray]] = {}
    for seed in retrieval.training.seeds:
        run_dir = retrieval.runs_dir / "graphsage" / name / f"seed{seed}"
        validate_run(run_dir, retrieval, seed)
        inputs += [run_dir / SETTINGS_FILE, run_dir / f"{GRAPH_VECTORS}.npy"]
        table = load_table(run_dir, GRAPH_VECTORS)
        graph = {split: graph_vector_scores(split_data, table) for split, split_data in data.items()}
        fixed_ranks: dict[Split, np.ndarray] = {}
        for feature, n_buckets in plans:
            key = _key(feature, n_buckets)
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
            if n_buckets == 1:
                _check_official(fusion, benchmark, seed, str(benchmark_path))
            result: dict[str, Any] = {"fit": fusion.to_dict()}
            for split in EVAL_SPLITS:
                fused, buckets = adaptive_scores(fusion, clip[split], graph[split], data[split].query_ids)
                ranks = ranks_of_relevant(fused, data[split].relevant_columns)
                if n_buckets == 1:
                    fixed_ranks[split] = ranks
                result[split.value] = {
                    "metrics": retrieval_metrics(ranks, grid.recall_ks),
                    "buckets": _bucket_rows(ranks, fixed_ranks[split], buckets, fusion),
                }
                if split is Split.TEST:
                    test_ranks.setdefault(key, []).append(ranks)
            entry = entries.setdefault(
                key, {"feature": None if n_buckets == 1 else feature.value, "n_buckets": n_buckets, "seeds": {}}
            )
            entry["seeds"][str(seed)] = result
            log.info(
                "seed %d %-16s buckets %d  val R@1 %.4f  test R@1 %.4f",
                seed,
                key,
                len(fusion.settings),
                result[Split.VAL.value]["metrics"]["recall@1"],
                result[Split.TEST.value]["metrics"]["recall@1"],
            )

    # Paired comparison with the fixed baseline: same queries, same seeds, captions clustered by image.
    test = data[Split.TEST]
    true_images = np.asarray(test.image_ids)[test.relevant_columns]
    fixed = np.stack(test_ranks[FIXED])
    for key, entry in entries.items():
        entry["summary"] = _summary(entry, retrieval.training.seeds)
        if key != FIXED:
            entry["paired_vs_fixed_test"] = paired_analysis(
                np.stack(test_ranks[key]),
                fixed,
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
                "selection": "per bucket on validation with the official alpha/top_k sweep and tie-break",
                "buckets": "edges at quantiles of the validation feature; test queries use the validation edges",
                "fixed": "one bucket; reproduces the benchmark's selection for every seed",
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
