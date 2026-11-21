#!/usr/bin/env python3
"""Train AdaMF-MAT on the MKG-W train triples; score link prediction and image retrieval.

Run from the repository root after `embed_mkg_w`:
    python -m scripts.train_mkg_w_adamf_mat                      # AdaMF-MAT (full)
    python -m scripts.train_mkg_w_adamf_mat --no-mat             # AdaMF w/o MAT
    python -m scripts.train_mkg_w_adamf_mat --mean-fusion        # w/o adaptive fusion
    python -m scripts.train_mkg_w_adamf_mat --modalities structure   # plain RotatE
    python -m scripts.train_mkg_w_adamf_mat --variant shuffled_edges # control graph

Two evaluations of each trained model:
- link prediction on the test triples over all entities (the paper's protocol);
- retrieval: a relation query gives its anchor entity, relation and direction, and every
  gallery entity is scored as the missing end of that triple. Name and description queries
  have no anchor and get a zero graph score, so CLIP alone ranks them.

Per-seed files go to <runs_dir>/adamf_mat/<run name>/seed-<n>/ and the summary over seeds
to <runs_dir>/adamf_mat/<run name>/benchmark.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from src.core.exceptions import AppError, ConfigError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.schemas import Split
from src.embeddings.store import EmbeddingTable, load_table, verify_inputs_unchanged
from src.evaluation.benchmark import aggregate_runs
from src.evaluation.link_prediction import known_completions, link_prediction_metrics
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.models.kg_adamf_mat import MODALITIES, KgAdamfMat
from src.models.kg_adamf_train import fit_kg_adamf, score_fn
from src.models.kg_experiment import LOGGED_PREFIXES, Variant, seed_summary, variant_triples
from src.models.train import seed_everything
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("train_mkg_w_adamf_mat")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"


def entity_image_vectors(entities: list[dict[str, Any]], images: EmbeddingTable) -> torch.Tensor:
    """Mean CLIP image vector of every entity; a zero row for an entity without images."""
    vectors = np.zeros((len(entities), images.vectors.shape[1]), dtype=np.float32)
    for row, entity in enumerate(entities):
        if entity["image_files"]:
            vectors[row] = images.select(entity["image_files"]).mean(axis=0)
    return torch.from_numpy(vectors)


def retrieval_scores(model: KgAdamfMat, data: EntitySplit, gallery: np.ndarray, batch_size: int) -> np.ndarray:
    """(queries × gallery) graph scores, each anchored row scaled to [0, 1]; zero rows for queries without an anchor."""
    score = score_fn(model)
    rows = [row for row, q in enumerate(data.queries) if q["anchor_id"] is not None]
    scores = np.zeros(data.clip_scores.shape, dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        queries = [data.queries[row] for row in batch]
        raw = score(np.array([q["anchor_id"] for q in queries]), np.array([q["relation_id"] for q in queries]),
                    np.array([q["direction"] == "tail" for q in queries]))[:, gallery]
        low, high = raw.min(axis=1, keepdims=True), raw.max(axis=1, keepdims=True)
        scores[batch] = (raw - low) / np.maximum(high - low, 1e-12)
    return scores


def run(config: MkgwRetrievalConfig, variant: Variant, seeds: tuple[int, ...]) -> None:
    """Train one model per seed on the variant's triples, evaluate, and write the summary."""
    if variant is Variant.NO_EDGES:
        raise ConfigError("a link-prediction model cannot train without triples")
    data = MkgwConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    queries = read_json(processed / "queries.json")
    all_triples = read_json(processed / "triples.json")
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    text_vectors = torch.from_numpy(load_table(embeddings_dir, "descriptions").select(
        [e["entity_id"] for e in entities]))
    image_vectors = entity_image_vectors(entities, image_table)

    gallery = np.array(list(gallery_columns(entities)))
    if max(config.top_k_grid) > len(gallery):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery)} entities")
    train = variant_triples(variant, all_triples["train"], config.path_shuffle_seed)
    val, test = all_triples[Split.VAL.value], all_triples[Split.TEST.value]
    known = [*all_triples["train"], *val, *test]
    completions = known_completions(known)
    splits = {s: build_entity_split(s, entities, queries, image_table, query_table) for s in (Split.VAL, Split.TEST)}
    training = config.adamf_training
    run_name = config.adamf_mat.run_name() + ("" if variant is Variant.REAL else f"-{variant.value}")
    out_dir = config.runs_dir / "adamf_mat" / run_name
    log.info("run=%s train triples=%d val=%d test=%d gallery=%d", run_name, len(train), len(val), len(test),
             len(gallery))

    max_hops = data.rules.max_hops
    per_seed: dict[str, Any] = {}
    link: list[dict[str, float]] = []
    reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    for seed in seeds:
        log.info("seed %d", seed)
        seed_everything(seed)
        model = KgAdamfMat(config.adamf_mat, len(entities), len(relations), image_vectors, text_vectors)
        result = fit_kg_adamf(model, train, val, known, training, seed)
        model.load_state_dict(result.state_dict)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result.state_dict, seed_dir / "model.pt")
        write_json(seed_dir / "history.json", result.history)

        test_link = link_prediction_metrics(score_fn(model), test, completions, training.eval_batch_size)
        link.append(test_link)
        log.info("seed %d best epoch %d, test link prediction: %s", seed, result.best_epoch,
                 {k: round(v, 4) for k, v in test_link.items()})
        scores = {s: retrieval_scores(model, d, gallery, training.eval_batch_size) for s, d in splits.items()}
        selected = select_alpha_top_k(splits[Split.VAL], scores[Split.VAL], config.alpha_grid, config.top_k_grid)
        per_seed[str(seed)] = {"best_epoch": result.best_epoch, "stop_reason": result.stop_reason.value,
                               "selected": selected}
        log.info("seed %d selected on val: %s", seed, selected)
        for split, split_data in splits.items():
            fused = rerank_scores(split_data.clip_scores, scores[split], selected["alpha"], int(selected["top_k"]))
            reranked[split].append(segment_metrics(split_data, fused, relations, max_hops, config.recall_ks))
            graph_only[split].append(segment_metrics(split_data, scores[split], relations, max_hops, config.recall_ks))

    report: dict[str, Any] = {
        "run_name": run_name,
        "variant": variant.value,
        "model_name": config.clip.model_name,
        "graph_triples": len(train),
        "seeds": per_seed,
        "link_prediction": aggregate_runs(link),
        "splits": {s.value: {"reranked": seed_summary(reranked[s]), "graph_only": seed_summary(graph_only[s])}
                   for s in splits},
    }
    write_json(out_dir / "benchmark.json", report)
    log.info("test link prediction over seeds: %s",
             {k: round(v["mean"], 4) for k, v in report["link_prediction"].items()})
    for split in splits:
        for name, row in report["splits"][split.value]["reranked"].items():
            if name.startswith(LOGGED_PREFIXES):
                alone = report["splits"][split.value]["graph_only"][name]
                log.info("%-5s %-26s n=%5d rerank R@1 %.3f±%.3f R@10 %.3f MRR %.3f | graph only R@1 %.3f R@10 %.3f",
                         split.value, name, row["answers"], row["recall@1"]["mean"], row["recall@1"]["std"],
                         row["recall@10"]["mean"], row["mrr"]["mean"],
                         alone["recall@1"]["mean"], alone["recall@10"]["mean"])
    log.info("wrote %s", out_dir / "benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variant", default=Variant.REAL.value,
                        choices=[Variant.REAL.value, Variant.SHUFFLED_EDGES.value])
    parser.add_argument("--no-mat", action="store_true", help="train without the modality generators")
    parser.add_argument("--mean-fusion", action="store_true", help="average the modalities instead of AdaMF weights")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, help="modalities to fuse; needs --no-mat "
                        "unless all three are kept")
    parser.add_argument("--seeds", nargs="+", type=int, help="subset of training.seeds, e.g. for a smoke run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = MkgwRetrievalConfig.from_yaml(args.config)
        modalities = tuple(m for m in MODALITIES if m in (args.modalities or MODALITIES))
        model_config = replace(
            config.adamf_mat,
            mat=config.adamf_mat.mat and not args.no_mat and modalities == MODALITIES,
            adaptive_fusion=config.adamf_mat.adaptive_fusion and not args.mean_fusion,
            modalities=modalities,
        )
        seeds = tuple(args.seeds or config.training.seeds)
        if set(seeds) - set(config.training.seeds):
            raise ConfigError(f"--seeds must be among training.seeds {config.training.seeds}")
        run(replace(config, adamf_mat=model_config), Variant(args.variant), seeds)
    except AppError:
        log.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
