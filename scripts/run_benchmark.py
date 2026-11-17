#!/usr/bin/env python3
"""Score a retrieval method on the validation and test splits of VG ∩ COCO.

Run from the repository root after `embed_vg_coco`:
    python -m scripts.run_benchmark --method clip
    python -m scripts.run_benchmark --method graph_lexical

Writes <runs_dir>/benchmark/<method>.json. Graph methods are reported on their own and fused
with CLIP at the alpha and top_k selected on validation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import META_FILE, load_table, verify_inputs_unchanged
from src.evaluation.benchmark import SplitData, build_split_data, clip_scores, score_method
from src.retrieval.config import RetrievalConfig
from src.retrieval.lexical import lexical_graph_scores
from src.retrieval.rerank import evaluate_reranking

log = logging.getLogger("run_benchmark")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
EVAL_SPLITS = (Split.VAL, Split.TEST)
_METRIC_DECIMALS = 4
GRAPH_DIR = "graph"
GRAPH_FILES = ("image_graphs", "concepts", "predicates")


class Method(StrEnum):
    """Retrieval methods without training or sampling: one run gives the exact result."""

    CLIP = "clip"
    GRAPH_LEXICAL = "graph_lexical"                  # caption words vs concept and predicate words
    GRAPH_LEXICAL_OBJECTS = "graph_lexical_objects"  # caption words vs concept words only


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


def run(config: RetrievalConfig, method: Method) -> None:
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
        "clip": {
            split.value: _rounded(score_method(clip[split], d, benchmark.recall_ks)) for split, d in data.items()
        },
    }
    inputs = [data_config.paths.split_file, embeddings_dir / META_FILE]
    for split in EVAL_SPLITS:
        log.info("clip          %-4s %s", split.value, result["clip"][split.value])

    if method is not Method.CLIP:
        graph, graph_files = _lexical_graph_scores(
            processed, queries, data, use_predicates=method is Method.GRAPH_LEXICAL
        )
        evaluation = evaluate_reranking(
            clip, graph, data, benchmark.alpha_grid, benchmark.top_k_grid, benchmark.recall_ks
        )
        result["graph_only"] = {s.value: _rounded(m) for s, m in evaluation.graph_only.items()}
        result["selected_on_validation"] = {"alpha": evaluation.selected.alpha, "top_k": evaluation.selected.top_k}
        result["reranked"] = {s.value: _rounded(m) for s, m in evaluation.reranked.items()}
        result["validation_sweep"] = [
            {"alpha": p.alpha, "top_k": p.top_k, **_rounded(p.metrics)} for p in evaluation.sweep
        ]
        inputs += graph_files
        log.info("selected on validation: alpha=%s top_k=%d", evaluation.selected.alpha, evaluation.selected.top_k)
        for split in EVAL_SPLITS:
            log.info("graph only    %-4s %s", split.value, result["graph_only"][split.value])
            log.info("clip + graph  %-4s %s", split.value, result["reranked"][split.value])

    result["inputs_sha256"] = {str(p): sha256_of(p) for p in inputs}
    out_file = config.runs_dir / "benchmark" / f"{method.value}.json"
    write_json(out_file, result)
    log.info("wrote %s", out_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True, choices=[m.value for m in Method])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(RetrievalConfig.from_yaml(args.config), Method(args.method))
    except AppError:
        log.exception("benchmark failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
