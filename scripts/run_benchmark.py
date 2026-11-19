#!/usr/bin/env python3
"""Score a retrieval method on the validation and test splits of VG ∩ COCO.

Run from the repository root after `embed_vg_coco`:
    python -m scripts.run_benchmark --method clip
    python -m scripts.run_benchmark --method graph_lexical
    python -m scripts.run_benchmark --method graphsage            # after train_graphsage
    python -m scripts.run_benchmark --method adamf_mat            # after train_graphsage --model adamf_mat

Writes <runs_dir>/benchmark/<method>.json. Graph methods are reported on their own and fused
with CLIP at the alpha and top_k selected on validation. `graphsage` reads the graph vectors
of every trained seed and reports each seed plus mean and standard deviation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import META_FILE, load_table, verify_inputs_unchanged
from src.evaluation.benchmark import (
    SplitData,
    aggregate_runs,
    build_split_data,
    clip_scores,
    graph_vector_scores,
    score_method,
)
from src.models.checkpoint import GRAPH_VECTORS, RESULT_FILE
from src.models.run_validation import validate_run
from src.retrieval.config import RetrievalConfig
from src.retrieval.lexical import lexical_graph_scores
from src.retrieval.rerank import RerankEvaluation, evaluate_reranking

log = logging.getLogger("run_benchmark")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
EVAL_SPLITS = (Split.VAL, Split.TEST)
_METRIC_DECIMALS = 4
GRAPH_DIR = "graph"
GRAPH_FILES = ("image_graphs", "concepts", "predicates")


class Method(StrEnum):
    """Retrieval methods. Those without training or sampling are exact in one run."""

    CLIP = "clip"
    GRAPH_LEXICAL = "graph_lexical"  # caption words vs concept and predicate words
    GRAPH_LEXICAL_OBJECTS = "graph_lexical_objects"  # caption words vs concept words only
    GRAPHSAGE = "graphsage"  # learned graph vectors, one run per seed
    ADAMF_MAT = "adamf_mat"  # adaptive fusion of graph, image and label text, one run per seed
    GRAPHSAGE_LABEL_TEXT = "graphsage_label_text"  # GraphSAGE + label-text branch, control for AdaMF-MAT


# Methods whose graph vectors come from training runs; the value is also their runs sub-directory.
TRAINED_METHODS = (Method.GRAPHSAGE, Method.ADAMF_MAT, Method.GRAPHSAGE_LABEL_TEXT)


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, _METRIC_DECIMALS) for name, value in metrics.items()}


def _lexical_graph_scores(
    processed: Path, queries: list[dict[str, Any]], data: dict[Split, SplitData], *, use_predicates: bool
) -> tuple[dict[Split, np.ndarray], list[Path]]:
    """Word-overlap graph score per split, and the graph files it was computed from."""
    files = {name: processed / GRAPH_DIR / f"{name}.json" for name in GRAPH_FILES}
    scores = lexical_graph_scores(
        read_json(files["image_graphs"]),
        {c["concept_id"]: c["text"] for c in read_json(files["concepts"])},
        {p["predicate_id"]: p["text"] for p in read_json(files["predicates"])},
        {q["query_id"]: q["text"] for q in queries},
        data,
        use_predicates=use_predicates,
    )
    return scores, list(files.values())


def _evaluation_report(evaluation: RerankEvaluation) -> dict[str, Any]:
    """Graph-only and fused metrics of one evaluation, with the validation-selected setting."""
    return {
        "selected_on_validation": {"alpha": evaluation.selected.alpha, "top_k": evaluation.selected.top_k},
        "graph_only": {s.value: _rounded(m) for s, m in evaluation.graph_only.items()},
        "reranked": {s.value: _rounded(m) for s, m in evaluation.reranked.items()},
    }


def _log_evaluation(prefix: str, evaluation: RerankEvaluation) -> None:
    log.info(
        "%sselected on validation: alpha=%s top_k=%d", prefix, evaluation.selected.alpha, evaluation.selected.top_k
    )
    for split in EVAL_SPLITS:
        log.info("%sgraph only    %-4s %s", prefix, split.value, _rounded(evaluation.graph_only[split]))
        log.info("%sclip + graph  %-4s %s", prefix, split.value, _rounded(evaluation.reranked[split]))


def _trained_seeds(
    config: RetrievalConfig,
    method: Method,
    name: str,
    clip: dict[Split, np.ndarray],
    data: dict[Split, SplitData],
    sweep: tuple[Any, Any, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluate every trained seed of a run; alpha and top_k are selected per seed on validation."""
    evaluations: dict[int, RerankEvaluation] = {}
    inputs: list[Path] = []
    for seed in config.training.seeds:
        run_dir = config.runs_dir / method.value / name / f"seed{seed}"
        if not (run_dir / RESULT_FILE).is_file():
            raise DataError(f"seed {seed} of run '{name}' has not finished training: {run_dir / RESULT_FILE} not found")

        # Reject a run with another intervention or stale training inputs.
        validate_run(
            run_dir,
            config,
            seed,
        )
        graph_table = load_table(run_dir, GRAPH_VECTORS)
        graph = {split: graph_vector_scores(split_data, graph_table) for split, split_data in data.items()}
        evaluations[seed] = evaluate_reranking(clip, graph, data, *sweep)
        inputs.extend(
            run_dir / entry for entry in (RESULT_FILE, "config.yaml", "graph_vectors.npy", "graph_vectors.ids.json")
        )
        _log_evaluation(f"seed {seed}: ", evaluations[seed])

    def summary(kind: str) -> dict[str, Any]:
        return {
            split.value: {
                metric: {key: round(value, _METRIC_DECIMALS) for key, value in stats.items()}
                for metric, stats in aggregate_runs([getattr(e, kind)[split] for e in evaluations.values()]).items()
            }
            for split in EVAL_SPLITS
        }

    report = {
        "seeds": {str(seed): _evaluation_report(evaluation) for seed, evaluation in evaluations.items()},
        "graph_only": summary("graph_only"),
        "reranked": summary("reranked"),
    }
    for split in EVAL_SPLITS:
        log.info("mean ± std  graph only   %-4s %s", split.value, report["graph_only"][split.value])
        log.info("mean ± std  clip + graph %-4s %s", split.value, report["reranked"][split.value])
    return report, inputs


def run(config: RetrievalConfig, method: Method, run_name: str | None = None) -> None:
    """Score `method` on each evaluation split and write the result file."""
    data_config = DataConfig.from_yaml(config.data_config)
    processed = data_config.paths.processed_dir
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    benchmark = config.benchmark

    data = {split: build_split_data(split, images, queries, image_table, query_table) for split in EVAL_SPLITS}
    clip = {split: clip_scores(split_data) for split, split_data in data.items()}
    result: dict[str, Any] = {
        "method": method.value,
        "clip_model": config.clip.model_name,
        "deterministic": True,
        "recall_ks": list(benchmark.recall_ks),
        "splits": {
            split.value: {"gallery_images": len(d.image_ids), "queries": len(d.query_ids)} for split, d in data.items()
        },
        "clip": {split.value: _rounded(score_method(clip[split], d, benchmark.recall_ks)) for split, d in data.items()},
    }
    inputs = [data_config.paths.split_file, embeddings_dir / META_FILE]
    for split in EVAL_SPLITS:
        log.info("clip          %-4s %s", split.value, result["clip"][split.value])

    sweep = (benchmark.alpha_grid, benchmark.top_k_grid, benchmark.recall_ks)
    if method in TRAINED_METHODS:
        if run_name is None and method is not Method.ADAMF_MAT and method is not Method.GRAPHSAGE:
            raise DataError(f"--run is required with --method {method.value}")
        default_name = config.adamf_mat.run_name() if method is Method.ADAMF_MAT else config.graphsage.run_name()
        name = run_name or default_name
        seeds, seed_inputs = _trained_seeds(config, method, name, clip, data, sweep)
        result |= {"run_name": name, "deterministic": False, **seeds}
        inputs += seed_inputs
    elif method is not Method.CLIP:
        graph, graph_files = _lexical_graph_scores(
            processed, queries, data, use_predicates=method is Method.GRAPH_LEXICAL
        )
        evaluation = evaluate_reranking(clip, graph, data, *sweep)
        result |= _evaluation_report(evaluation)
        result["validation_sweep"] = [
            {"alpha": p.alpha, "top_k": p.top_k, **_rounded(p.metrics)} for p in evaluation.sweep
        ]
        inputs += graph_files
        _log_evaluation("", evaluation)

    result["inputs_sha256"] = {str(p): sha256_of(p) for p in inputs}
    suffix = f"-{result['run_name']}" if method in TRAINED_METHODS else ""
    out_file = config.runs_dir / "benchmark" / f"{method.value}{suffix}.json"
    write_json(out_file, result)
    log.info("wrote %s", out_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True, choices=[m.value for m in Method])
    parser.add_argument(
        "--run", help="training run to score with --method graphsage or adamf_mat; default: the config's architecture"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(RetrievalConfig.from_yaml(args.config), Method(args.method), args.run)
    except AppError:
        log.exception("benchmark failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
