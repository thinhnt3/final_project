#!/usr/bin/env python3
"""Per-image AdaMF fusion weights of trained AdaMF-MAT runs, and how much each modality really counts.

    python -m scripts.adamf_weight_stats --run adamf-mat-d256-mu1 --run adamf-d256-l2

For every seed, on the validation images: mean, standard deviation and coefficient of
variation (std / mean) of each modality's weight across images, the mean norm of each
modality embedding, and the effective share w·‖e‖ / Σ w·‖e‖. A near-zero coefficient of
variation means the weights do not adapt to the image: the fusion is a fixed weighted sum.
Unnormalized runs can offset a small weight by a large norm, so read the effective share.
Writes <runs_dir>/analysis/adamf_weights-<run>.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import torch
from torch_geometric.data import Batch

from src.core.exceptions import AppError
from src.data.json_io import write_json
from src.data.schemas import Split
from src.models.run_loading import load_graph_inputs, load_trained_encoder
from src.models.train_data import build_retrieval_set
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("adamf_weight_stats")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
BATCH_SIZE = 512
_DECIMALS = 5


@torch.no_grad()
def weight_stats(model: Any, graphs: list[Any]) -> dict[str, dict[str, float]]:
    """Statistics of the fusion weights and embedding norms over `graphs`, per modality."""
    weights, norms = [], []
    for start in range(0, len(graphs), BATCH_SIZE):
        embeddings = model.modality_embeddings(Batch.from_data_list(graphs[start:start + BATCH_SIZE]))
        weights.append(model.modality_weights(*embeddings))
        norms.append(torch.stack([e.norm(dim=-1) for e in embeddings], dim=1))
    w, n = torch.cat(weights), torch.cat(norms)
    effective = w * n
    share = effective / effective.sum(dim=1, keepdim=True)
    result = {}
    for column, modality in enumerate(model.config.modalities):
        mean, std = w[:, column].mean().item(), w[:, column].std().item()
        result[modality] = {
            "weight_mean": mean,
            "weight_std": std,
            "weight_cv": std / mean if mean > 0 else float("inf"),
            "weight_min": w[:, column].min().item(),
            "weight_max": w[:, column].max().item(),
            "norm_mean": n[:, column].mean().item(),
            "effective_share_mean": share[:, column].mean().item(),
        }
    return {m: {k: round(v, _DECIMALS) for k, v in s.items()} for m, s in result.items()}


def main() -> int:
    """Parse the command line, compute the statistics of every requested run, write one file per run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="append", required=True, help="AdaMF-MAT run name; repeatable")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        inputs = load_graph_inputs(config)
        val, _ = build_retrieval_set(
            Split.VAL, inputs.images, inputs.queries, inputs.image_graphs, inputs.image_table, inputs.query_table
        )
        for name in args.run:
            report: dict[str, Any] = {
                "run_name": name, "split": Split.VAL.value, "images": len(val.graphs), "seeds": {}
            }
            for seed in config.training.seeds:
                run_dir = config.runs_dir / "adamf_mat" / name / f"seed{seed}"
                model = load_trained_encoder(run_dir, "adamf_mat", inputs)
                report["seeds"][str(seed)] = weight_stats(model, list(val.graphs))
                for modality, stats in report["seeds"][str(seed)].items():
                    log.info("%s seed %d %-9s %s", name, seed, modality, stats)
            out = config.runs_dir / "analysis" / f"adamf_weights-{name}.json"
            write_json(out, report)
            log.info("wrote %s", out)
    except AppError:
        log.exception("weight statistics failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
