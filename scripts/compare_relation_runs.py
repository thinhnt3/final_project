#!/usr/bin/env python3
"""Paired bootstrap of relation-query accuracy between two scored settings (from --save-ranks files).

    python -m scripts.compare_relation_runs runs/analysis/relation_queries-suite.ranks.npz \\
        --a "adamf_mat/adamf-d256-l2|graph_only" --b "adamf_mat/adamf-d256-l2-so0.3|graph_only" --group contrast

Seeds stay paired (same seed index in both), queries are resampled by relevant image.
Reports the accuracy difference A − B in percentage points with a 95 % interval. A
setting can also be compared with its masked version (`...|graph_only_masked`).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.evaluation.ablation_analysis import paired_analysis


def main() -> int:
    """Print the paired analysis as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ranks", help="a .ranks.npz written by eval_relation_queries --save-ranks")
    parser.add_argument("--a", required=True, help="method|setting")
    parser.add_argument("--b", required=True, help="method|setting")
    parser.add_argument("--group", default="all", choices=["all", "generic", "specific", "contrast"])
    args = parser.parse_args()
    data = np.load(args.ranks)
    a, b = data[f"{args.a}|{args.group}"], data[f"{args.b}|{args.group}"]
    if a.shape[0] != b.shape[0]:
        # CLIP has one "seed"; repeat it to pair with each trained seed.
        a, b = (np.repeat(a, b.shape[0], axis=0), b) if a.shape[0] == 1 else (a, np.repeat(b, a.shape[0], axis=0))
    result = paired_analysis(a, b, data[f"image_ids|{args.group}"])
    print(json.dumps({"a": args.a, "b": args.b, "group": args.group, **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
