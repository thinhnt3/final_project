#!/usr/bin/env python3
"""Score CLIP and trained encoders on indirect queries (a hypernym names the subject) and their direct twins.

    python -m scripts.build_kg_variant      # once
    python -m scripts.eval_indirect_queries --base-run adamf-d256-l2 --kg-run adamf-d256-l2

Base runs (trained on graphs without IsA edges) see the base graphs; KG runs (configs/
retrieval/vg_coco_kg.yaml) see the graphs with IsA edges, and are scored a second time with
the relevant image's IsA edges removed (`kg_masked`): the drop is how much they rely on
those edges. Reported for each kind: Recall@K over the 1,500 test images, and top-1
accuracy among the relevant image and its confusers. Alpha and top_k are the ones each
seed selected on validation captions. Writes <runs_dir>/analysis/indirect_queries.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable, load_table
from src.evaluation.benchmark import SplitData, aggregate_runs, build_split_data, clip_scores, graph_vector_scores
from src.evaluation.metrics import ranks_of_relevant, retrieval_metrics
from src.evaluation.relation_queries import chance_accuracy, confuser_ranks
from src.models.graph_data import image_graph_to_data
from src.models.run_loading import GraphInputs, load_graph_inputs, load_trained_encoder
from src.models.train import encode_graphs
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("eval_indirect_queries")

BASE_CONFIG = "configs/retrieval/vg_coco.yaml"
KG_CONFIG = "configs/retrieval/vg_coco_kg.yaml"
KINDS = ("indirect", "direct")
_DECIMALS = 4


def _metrics(
    scores: np.ndarray, data: SplitData, confusers: Sequence[Sequence[int]], ks: Sequence[int]
) -> dict[str, float]:
    full = retrieval_metrics(ranks_of_relevant(scores, data.relevant_columns), ks)
    among = confuser_ranks(scores, data.relevant_columns.tolist(), confusers)
    return {**full, "confuser_acc@1": float((among == 1).mean())}


def _encode(
    model: Any, inputs: GraphInputs, ids: Sequence[int], graph_of: dict[int, Any], config: Any
) -> EmbeddingTable:
    row = inputs.image_table.row_of()
    graphs = [image_graph_to_data(graph_of[i], row[i]) for i in ids]
    return EmbeddingTable(ids=tuple(ids), vectors=encode_graphs(model, graphs, config.training).numpy())


def main() -> int:
    """Score every requested run and seed on both query kinds, write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-run", action="append", default=[], help="AdaMF run trained without IsA edges")
    parser.add_argument("--kg-run", action="append", default=[], help="AdaMF run trained with IsA edges")
    parser.add_argument("--out", default="indirect_queries")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        base, kg = RetrievalConfig.from_yaml(BASE_CONFIG), RetrievalConfig.from_yaml(KG_CONFIG)
        ks = base.benchmark.recall_ks
        base_inputs, kg_inputs = load_graph_inputs(base), load_graph_inputs(kg)
        kg_dir = DataConfig.from_yaml(kg.data_config).paths.processed_dir
        queries = read_json(kg_dir / "graph" / "indirect_queries.json")
        table = load_table(kg.model_embeddings_dir(), "indirect_queries")
        base_graph = {int(g["vg_image_id"]): g for g in base_inputs.image_graphs}
        kg_graph = {int(g["vg_image_id"]): g for g in kg_inputs.image_graphs}

        kinds: dict[str, tuple[SplitData, list[list[int]]]] = {}
        for kind in KINDS:
            selected = sorted((q for q in queries if q["kind"] == kind), key=lambda q: q["query_id"])
            data = build_split_data(Split.TEST, base_inputs.images, selected, base_inputs.image_table, table)
            column = {image_id: c for c, image_id in enumerate(data.image_ids)}
            kinds[kind] = (data, [[column[int(i)] for i in q["confuser_ids"]] for q in selected])
        test_ids = kinds["indirect"][0].image_ids

        results: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {}

        def record(method: str, setting: str, scores_of: dict[str, np.ndarray]) -> None:
            for kind, scores in scores_of.items():
                data, confusers = kinds[kind]
                results.setdefault(method, {}).setdefault(setting, {}).setdefault(kind, []).append(
                    _metrics(scores, data, confusers, ks)
                )

        record("clip", "clip", {k: clip_scores(kinds[k][0]) for k in KINDS})
        plan = [("base", name, base, base_inputs, base_graph) for name in args.base_run]
        plan += [("kg", name, kg, kg_inputs, kg_graph) for name in args.kg_run]
        for variant, run, config, inputs, graph_of in plan:
            method = f"{variant}/{run}"
            selected = read_json(config.runs_dir / "benchmark" / f"adamf_mat-{run}.json")["seeds"]
            for seed in config.training.seeds:
                choice = selected[str(seed)]["selected_on_validation"]
                model = load_trained_encoder(config.runs_dir / "adamf_mat" / run / f"seed{seed}", "adamf_mat", inputs)
                vectors = _encode(model, inputs, test_ids, graph_of, config)
                settings = {"": vectors}
                if variant == "kg":
                    # The relevant image of every query loses its IsA edges (its base graph).
                    relevant = sorted({int(q["relevant_ids"][0]) for q in queries})
                    unmasked = dict(zip(vectors.ids, vectors.vectors, strict=True))
                    masked_vectors = _encode(model, inputs, relevant, base_graph, config)
                    unmasked.update(zip(masked_vectors.ids, masked_vectors.vectors, strict=True))
                    settings["_masked"] = EmbeddingTable(
                        ids=test_ids, vectors=np.stack([unmasked[i] for i in test_ids]).astype(np.float32)
                    )
                for suffix, graph_table in settings.items():
                    graph = {k: graph_vector_scores(kinds[k][0], graph_table) for k in KINDS}
                    record(method, f"graph_only{suffix}", graph)
                    record(method, f"reranked{suffix}", {
                        k: rerank_scores(clip_scores(kinds[k][0]), graph[k], choice["alpha"], choice["top_k"])
                        for k in KINDS
                    })
                log.info("scored %s seed %d", method, seed)

        report = {
            "split": Split.TEST.value,
            "queries": {k: len(kinds[k][1]) for k in KINDS},
            "confuser_chance_acc@1": round(chance_accuracy(kinds["indirect"][1]), _DECIMALS),
            "note": "_masked: the relevant image's graph without IsA edges, the others unchanged",
            "methods": {
                method: {
                    setting: {
                        kind: {
                            m: {k: round(v, _DECIMALS) for k, v in s.items()}
                            for m, s in aggregate_runs(runs).items()
                        }
                        for kind, runs in by_kind.items()
                    }
                    for setting, by_kind in settings_of.items()
                }
                for method, settings_of in results.items()
            },
        }
        out = Path(base.runs_dir) / "analysis" / f"{args.out}.json"
        write_json(out, report)
        log.info("wrote %s", out)
    except AppError:
        log.exception("indirect query evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
