#!/usr/bin/env python3
"""Summarize the VN heritage retrieval runs over several claim splits.

Run from the repository root after `run_vn_heritage_retrieval --split-seed <n>` for every seed:
    python -m scripts.summarize_vn_heritage_splits --split-seeds 1 2 3

Each split gives one number per method (its mean over training seeds). The summary reports
the mean and std of that number over splits, and paired differences between methods: the
same split is compared with itself, so the difficulty of a split cancels out.
Written to <runs_dir>/splits_summary.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.json_io import read_json, write_json
from src.retrieval.mkg_w_config import MkgwRetrievalConfig

log = logging.getLogger("summarize_vn_heritage_splits")

DEFAULT_CONFIG = "configs/retrieval/vn_heritage_core.yaml"
_CLIP = "clip"
_GRAPHS = ("no_edges", "claims", "claims_shuffled", "generic", "kind", "section", "cue", "cue_shuffled")
_SEGMENTS = ("all", "group=direct", "group=indirect", "hops=2", "hops=3", "hops=over_3_or_none",
             "density=sparse", "density=medium", "density=dense")
_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
# (method, baseline): the difference method - baseline is reported.
_COMPARISONS = (
    ("claims", _CLIP), ("claims", "no_edges"), ("claims", "claims_shuffled"),
    ("generic", "claims"), ("cue", "claims"), ("cue", "generic"), ("kind", "generic"),
    ("section", "generic"), ("cue", "cue_shuffled"),
)


def split_values(runs_dir: Path, split: str) -> dict[str, dict[str, dict[str, float]]]:
    """method -> segment -> metric for one claim split (reranked scores for the graph methods)."""
    clip = read_json(runs_dir / _CLIP / "benchmark.json")["splits"][split]["segments"]
    values = {_CLIP: {segment: {m: clip[segment][m] for m in _METRICS} for segment in _SEGMENTS}}
    for graph in _GRAPHS:
        rows = read_json(runs_dir / "graphsage" / graph / "benchmark.json")["splits"][split]["reranked"]
        values[graph] = {segment: {m: rows[segment][m]["mean"] for m in _METRICS} for segment in _SEGMENTS}
    return values


def paired(method: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """Mean difference over splits, its standard error and how many splits favour the method."""
    diff = method - baseline
    if diff.size < 2:
        raise DataError("a paired comparison needs at least two splits")
    return {
        "mean_diff": round(float(diff.mean()), 4),
        "std_error": round(float(diff.std(ddof=1) / np.sqrt(diff.size)), 4),
        "splits_better": int((diff > 0).sum()),
        "splits_worse": int((diff < 0).sum()),
    }


def summarize(runs_dir: Path, seeds: list[int], split: str) -> dict[str, Any]:
    """Per-method mean / std over claim splits and the paired comparisons."""
    per_split = [split_values(runs_dir / f"split-{seed}", split) for seed in seeds]

    def series(method: str, segment: str, metric: str) -> np.ndarray:
        return np.asarray([values[method][segment][metric] for values in per_split])

    methods = {
        method: {segment: {metric: {"mean": round(float(series(method, segment, metric).mean()), 4),
                                    "std": round(float(series(method, segment, metric).std(ddof=1)), 4)}
                           for metric in _METRICS} for segment in _SEGMENTS}
        for method in (_CLIP, *_GRAPHS)
    }
    comparisons = {
        f"{method} - {baseline}": {segment: {metric: paired(series(method, segment, metric),
                                                            series(baseline, segment, metric))
                                             for metric in _METRICS} for segment in _SEGMENTS}
        for method, baseline in _COMPARISONS
    }
    return {"split": split, "claim_split_seeds": seeds, "methods": methods, "paired": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        runs_dir = MkgwRetrievalConfig.from_yaml(args.config).runs_dir
        summary = summarize(runs_dir, args.split_seeds, args.split)
    except AppError:
        log.exception("summary failed")
        return 1
    write_json(runs_dir / f"splits_summary_{args.split}.json", summary)
    for segment in ("all", "group=indirect", "group=direct", "density=sparse", "density=medium", "density=dense"):
        log.info("\n== %s / %s: mean ± std over %d claim splits ==", args.split, segment, len(args.split_seeds))
        for method, rows in summary["methods"].items():
            log.info("%-16s %s", method, "  ".join(
                f"{m} {rows[segment][m]['mean']:.3f}±{rows[segment][m]['std']:.3f}" for m in _METRICS))
        log.info("-- paired difference (mean ± standard error, splits better/worse) --")
        for name, rows in summary["paired"].items():
            log.info("%-26s %s", name, "  ".join(
                f"{m} {rows[segment][m]['mean_diff']:+.3f}±{rows[segment][m]['std_error']:.3f} "
                f"({rows[segment][m]['splits_better']}/{rows[segment][m]['splits_worse']})" for m in _METRICS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
