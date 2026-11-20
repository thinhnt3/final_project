#!/usr/bin/env python3
"""How much a trained encoder depends on clean, human-annotated scene graphs. Eval only.

    python -m scripts.eval_graph_noise --method adamf_mat --run adamf-mat-d256-mu1
    python -m scripts.eval_graph_noise --method graphsage --run directed-clip_image-l3-h512

The test scene graphs are corrupted, then re-encoded by each seed's best checkpoint:
- drop_edge: each relation (subject, predicate, object) is removed with this probability;
- drop_node: each object is removed with this probability, with its relations;
- flip_label: each object's concept is replaced by a random concept absent from the image.
Every setting uses the same corruption of each image for every model (seeded by image id).
Reported per setting: the encoder alone, and CLIP + encoder with the alpha and top_k this
seed selected on clean validation data (runs/benchmark/<method>-<run>.json). Nothing is
tuned on test. Writes <runs_dir>/analysis/graph_noise-<method>-<run>.json.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.core.exceptions import AppError, DataError
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable
from src.evaluation.benchmark import aggregate_runs, build_split_data, clip_scores, graph_vector_scores, score_method
from src.models.ablations import Corruption, corrupt_graph
from src.models.graph_data import image_graph_to_data
from src.models.run_loading import load_graph_inputs, load_trained_encoder
from src.models.train import encode_graphs
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("eval_graph_noise")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
NOISE_SEED = 20261119
_DECIMALS = 4


# The grid of the experiment brief: edges, nodes, labels, one kind of noise at a time.
GRID = (
    Corruption(),
    Corruption(drop_edge=0.3),
    Corruption(drop_edge=0.5),
    Corruption(drop_node=0.3),
    Corruption(flip_label=0.2),
)

# Stand-in for predicted scene graphs (no scene-graph generator here): each kind of noise at
# several levels, and combinations where a detector misses objects, mislabels some and
# misses relations at the same time.
WIDE_GRID = (
    Corruption(),
    Corruption(drop_node=0.1), Corruption(drop_node=0.3), Corruption(drop_node=0.5),
    Corruption(flip_label=0.1), Corruption(flip_label=0.2), Corruption(flip_label=0.4),
    Corruption(drop_edge=0.5), Corruption(drop_edge=1.0),
    Corruption(drop_edge=0.3, drop_node=0.2, flip_label=0.1),
    Corruption(drop_edge=0.5, drop_node=0.4, flip_label=0.3),
)


def main() -> int:
    """Parse the command line, score every seed under every corruption setting, write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True, choices=["graphsage", "adamf_mat", "graphsage_label_text"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--grid", choices=["brief", "wide"], default="brief", help="wide: C4 stand-in for SGG noise")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        ks = config.benchmark.recall_ks
        inputs = load_graph_inputs(config)
        test = build_split_data(Split.TEST, inputs.images, inputs.queries, inputs.image_table, inputs.query_table)
        clip = clip_scores(test)
        selected = read_json(config.runs_dir / "benchmark" / f"{args.method}-{args.run}.json")["seeds"]
        graph_of = {int(g["vg_image_id"]): g for g in inputs.image_graphs}
        image_row = inputs.image_table.row_of()
        n_concepts = inputs.concepts.shape[0]

        per_setting: dict[str, dict[str, list[dict[str, float]]]] = {}
        for seed in config.training.seeds:
            choice = selected[str(seed)]["selected_on_validation"]
            run_dir = config.runs_dir / args.method / args.run / f"seed{seed}"
            model = load_trained_encoder(run_dir, args.method, inputs)
            for noise in WIDE_GRID if args.grid == "wide" else GRID:
                graphs = [
                    image_graph_to_data(corrupt_graph(graph_of[i], noise, n_concepts, NOISE_SEED), image_row[i])
                    for i in test.image_ids
                ]
                vectors = encode_graphs(model, graphs, config.training).numpy()
                graph = graph_vector_scores(test, EmbeddingTable(ids=test.image_ids, vectors=vectors))
                fused = rerank_scores(clip, graph, choice["alpha"], choice["top_k"])
                results = per_setting.setdefault(noise.name(), {"graph_only": [], "reranked": []})
                results["graph_only"].append(score_method(graph, test, ks))
                results["reranked"].append(score_method(fused, test, ks))
                log.info(
                    "seed %d %-14s graph only R@1 %.4f  clip + graph R@1 %.4f",
                    seed, noise.name(), results["graph_only"][-1]["recall@1"], results["reranked"][-1]["recall@1"],
                )
        if not per_setting:
            raise DataError("no seeds to score")
        report = {
            "method": args.method,
            "run_name": args.run,
            "split": Split.TEST.value,
            "noise_seed": NOISE_SEED,
            "settings": {
                name: {
                    kind: {m: {k: round(v, _DECIMALS) for k, v in s.items()} for m, s in aggregate_runs(runs).items()}
                    for kind, runs in kinds.items()
                }
                for name, kinds in per_setting.items()
            },
        }
        for name, kinds in report["settings"].items():
            log.info("%-14s graph only R@1 %s  clip + graph R@1 %s", name,
                     kinds["graph_only"]["recall@1"], kinds["reranked"]["recall@1"])
        suffix = "-wide" if args.grid == "wide" else ""
        out = config.runs_dir / "analysis" / f"graph_noise-{args.method}-{args.run}{suffix}.json"
        write_json(out, report)
        log.info("wrote %s", out)
    except AppError:
        log.exception("graph noise evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
