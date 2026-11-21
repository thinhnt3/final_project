#!/usr/bin/env python3
"""CLIP and CLIP + GraphSAGE text-to-image retrieval on the VN heritage core.

Run from the repository root after `build_vn_heritage_queries`:
    python -m scripts.run_vn_heritage_retrieval --stage embed
    python -m scripts.run_vn_heritage_retrieval --stage clip
    python -m scripts.run_vn_heritage_retrieval --stage graphsage --graphs claims cue
    python -m scripts.run_vn_heritage_retrieval --stage adamf --graphs claims cue
    python -m scripts.run_vn_heritage_retrieval --stage adamf --structure-only   # plain RotatE

Two galleries are scored (`--gallery`, both by default):
    image   every image; an image is correct when it shows an answer place. An image takes the
            graph score of the place it shows, i.e. it is reached over its image-shows-place edge.
    place   one entry per place, scored by its best image.
The encoder is trained once, on places, and its entity vectors serve both galleries.

A graph variant names the edges the encoder sees:
    no_edges         none: an MLP on each entity's own name vector
    claims           train claims only
    generic | kind | section | cue
                     train claims + article edges under that relation naming
    <name>_shuffled    the same edges with tails re-paired at random
    <name>_untyped     the same edges under one relation (no predicate labels)
    <name>_undirected  every edge also in the opposite direction

The adamf stage trains AdaMF-MAT, a link-prediction model, on the same edges. A relation query
gives its anchor entity, relation and direction, and every place is scored as the missing end
of that triple. Name queries have no anchor and get a zero graph score, so CLIP ranks them.

Results go to <runs_dir>/clip/, <runs_dir>/graphsage/<graph>/ and
<runs_dir>/adamf_mat/<model>/<graph>/: benchmark_images.json for the image gallery and
benchmark.json for the place gallery. The adamf stage scores places only.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.config_parsing import read_mapping, require_section
from src.core.exceptions import AppError, ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.data.vn_heritage_query_config import QueryConfig
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import (
    META_FILE,
    EmbeddingTable,
    encode_in_batches,
    load_table,
    save_table,
    table_paths,
    verify_inputs_unchanged,
)
from src.evaluation.benchmark import aggregate_runs
from src.evaluation.image_gallery import ImageSplit, build_image_split
from src.evaluation.link_prediction import known_completions, link_prediction_metrics
from src.evaluation.metrics import retrieval_metrics
from src.evaluation.mkg_w_benchmark import (
    EntitySplit,
    build_entity_split,
    gallery_columns,
    segment_metrics,
    select_alpha_top_k,
)
from src.models.kg_adamf_mat import KgAdamfMat, KgAdamfMatConfig
from src.models.kg_adamf_train import KgAdamfTrainingConfig, fit_kg_adamf, score_fn
from src.models.kg_graphsage import KgEntityEncoder
from src.models.kg_train import KgQuerySet, fit
from src.models.train import seed_everything
from src.retrieval.kg_adamf_config import load_adamf_configs
from src.retrieval.mkg_w_config import MkgwRetrievalConfig
from src.retrieval.path_score import shuffled_tails
from src.retrieval.rerank import rerank_scores

log = logging.getLogger("run_vn_heritage_retrieval")

DEFAULT_CONFIG = "configs/retrieval/vn_heritage_core.yaml"
_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
_NO_EDGES = "no_edges"
_CLAIMS = "claims"
_SHUFFLED = "_shuffled"
_UNTYPED = "_untyped"
_UNDIRECTED = "_undirected"
# Relation id given to every edge of an untyped graph.
_UNTYPED_RELATION = 0
_LOGGED_PREFIXES = ("all", "group=", "hops=", "density=")

IdTriple = tuple[int, int, int]


class Gallery(StrEnum):
    """What a query is ranked against."""

    IMAGE = "image"
    PLACE = "place"


@dataclasses.dataclass(frozen=True)
class GalleryGrid:
    """Search grids that depend on the size of the gallery."""

    top_ks: tuple[int, ...]        # candidates reranked
    candidate_ks: tuple[int, ...]  # extra recall cut-offs of the CLIP report


def report_name(gallery: Gallery) -> str:
    """File name of a gallery's report inside a run directory."""
    return "benchmark_images.json" if gallery is Gallery.IMAGE else "benchmark.json"


def gallery_grids(config: MkgwRetrievalConfig, raw: Mapping[str, Any], galleries: Sequence[Gallery],
                  ) -> dict[Gallery, GalleryGrid]:
    """Grids of each requested gallery: `benchmark` for places, `image_gallery` for images."""
    grids: dict[Gallery, GalleryGrid] = {}
    for gallery in galleries:
        if gallery is Gallery.PLACE:
            grids[gallery] = GalleryGrid(config.top_k_grid, config.candidate_ks)
            continue
        section = require_section(dict(raw), "image_gallery")
        values: dict[str, tuple[int, ...]] = {}
        for key in ("top_k_grid", "candidate_ks"):
            value = section.get(key)
            if (not isinstance(value, list) or not value
                    or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in value)):
                raise ConfigError(f"image_gallery.{key} must be a non-empty list of integers >= 1")
            values[key] = tuple(value)
        grids[gallery] = GalleryGrid(values["top_k_grid"], values["candidate_ks"])
    return grids


def embed(config: MkgwRetrievalConfig, data: QueryConfig, image_dir: Path) -> None:
    """Encode image files, query texts and entity names; write the tables and their meta."""
    inputs = {"entities": data.out_dir / "entities.json", "queries": data.out_dir / "queries.json"}
    entities = read_json(inputs["entities"])
    queries = read_json(inputs["queries"])
    image_ids = [file for entity in entities for file in entity["image_files"]]
    missing = [file for file in image_ids if not (image_dir / file).is_file()]
    if missing:
        raise DataError(f"{len(missing)} image files are missing in {image_dir}, e.g. {missing[:3]}")

    encoder = ClipEncoder(config.clip.model_name, config.clip.device)
    log.info("model=%s device=%s dim=%d", config.clip.model_name, encoder.device, encoder.dim)
    text_batch = config.clip.text_batch_size
    jobs: dict[str, tuple[list[Any], list[Any], Any, int]] = {
        "entities": ([e["entity_id"] for e in entities], [e["name"] for e in entities],
                     encoder.encode_texts, text_batch),
        "queries": ([q["query_id"] for q in queries], [q["text"] for q in queries], encoder.encode_texts, text_batch),
        "images": (image_ids, [image_dir / file for file in image_ids], encoder.encode_image_files,
                   config.clip.image_batch_size),
    }
    out_dir = config.model_embeddings_dir()
    tables = {}
    for name, (ids, items, encode, batch_size) in jobs.items():
        tables[name] = EmbeddingTable(ids=tuple(ids), vectors=encode_in_batches(items, encode, batch_size))
        save_table(out_dir, name, tables[name])
        log.info("%s: %s", name, tables[name].vectors.shape)
    write_json(out_dir / META_FILE, {
        "model_name": config.clip.model_name,
        "device": encoder.device,
        "dim": encoder.dim,
        "tables": {
            name: {"rows": len(table.ids), "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)}}
            for name, table in tables.items()
        },
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs.values()},
    })
    log.info("wrote %s", out_dir)


def _load_splits(
    config: MkgwRetrievalConfig, data: QueryConfig, gallery: Gallery = Gallery.PLACE
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[Split, EntitySplit]]:
    """Entities, relations, queries and the val / test splits scored by CLIP against a gallery."""
    entities = read_json(data.out_dir / "entities.json")
    relations = read_json(data.out_dir / "relations.json")
    queries = read_json(data.out_dir / "queries.json")
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    image_table = load_table(embeddings_dir, "images")
    query_table = load_table(embeddings_dir, "queries")
    build = build_image_split if gallery is Gallery.IMAGE else build_entity_split
    splits: dict[Split, EntitySplit] = {s: build(s, entities, queries, image_table, query_table)
                                        for s in (Split.VAL, Split.TEST)}
    return entities, relations, queries, splits


def _column_entities(split_data: EntitySplit, entities: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Entity id behind each gallery column: the place itself, or the place an image shows."""
    if isinstance(split_data, ImageSplit):
        return np.array(split_data.image_entities)
    return np.array(list(gallery_columns(entities)))


def _segments(
    split_data: EntitySplit, scores: np.ndarray, relations: Sequence[Mapping[str, Any]], data: QueryConfig,
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Report segments of the shared benchmark plus one per density tier of the indirect queries."""
    segments = segment_metrics(split_data, scores, relations, data.rules.max_hops, ks)
    strata = read_json(data.out_dir / "strata.json")
    ranks, rows = split_data.ranks(scores)
    pairs_of: dict[str, list[int]] = {}
    for pair, row in enumerate(rows):
        query_id = split_data.queries[row]["query_id"]
        if query_id in strata:
            pairs_of.setdefault(f"density={strata[query_id]['tier']}", []).append(pair)
    for name, pairs in sorted(pairs_of.items()):
        metrics = retrieval_metrics(ranks[pairs], ks)
        segments[name] = {"answers": len(pairs), **{k: round(v, 4) for k, v in metrics.items()}}
    return segments


def run_clip(config: MkgwRetrievalConfig, data: QueryConfig, gallery: Gallery, grid: GalleryGrid) -> None:
    """Rank the gallery for every val and test query with CLIP cosine; write the report."""
    entities, relations, _, splits = _load_splits(config, data, gallery)
    ks = sorted({*config.recall_ks, *grid.candidate_ks})
    report: dict[str, Any] = {"model_name": config.clip.model_name, "gallery": gallery.value,
                              "gallery_size": int(splits[Split.TEST].clip_scores.shape[1]), "splits": {}}
    for split, split_data in splits.items():
        segments = _segments(split_data, split_data.clip_scores, relations, data, ks)
        report["splits"][split.value] = {"queries": len(split_data.queries), "segments": segments}
        for name, row in segments.items():
            if name.startswith(_LOGGED_PREFIXES):
                log.info("%-5s %-5s %-26s n=%4d R@1 %.3f R@5 %.3f R@10 %.3f MRR %.3f", gallery.value, split.value,
                         name, row["answers"], row["recall@1"], row["recall@5"], row["recall@10"], row["mrr"])
    out_path = config.runs_dir / "clip" / report_name(gallery)
    write_json(out_path, report)
    log.info("wrote %s", out_path)


def graph_variant(name: str, triples: Mapping[str, Any], seed: int) -> tuple[list[IdTriple], dict[IdTriple, list[int]]]:
    """Edges of one graph variant and, for every fact it states, the rows of the edges stating it.

    See the module docstring for the names. In a shuffled graph the facts are the re-paired
    edges themselves: a true fact is no longer an edge there.
    """
    base, modifiers = name, set()
    while (suffix := next((m for m in (_SHUFFLED, _UNTYPED, _UNDIRECTED) if base.endswith(m)), None)) is not None:
        base = base.removesuffix(suffix)
        modifiers.add(suffix)
    if base == _NO_EDGES:
        return [], {}
    if base != _CLAIMS and base not in triples["articles"]:
        raise ConfigError(f"unknown graph {name!r}; article schemes are {sorted(triples['articles'])}")
    rows = triples[Split.TRAIN.value] + ([] if base == _CLAIMS else triples["articles"][base])
    graph = [(head, relation, tail) for head, relation, tail in rows]
    if _SHUFFLED in modifiers:
        pairs = shuffled_tails([(head, tail) for head, _, tail in graph], seed)
        graph = [(head, relation, tail) for (head, tail), (_, relation, _) in zip(pairs, graph, strict=True)]
    fact_rows = {fact: [row] for row, fact in enumerate(graph)}
    if _UNTYPED in modifiers:
        graph = [(head, _UNTYPED_RELATION, tail) for head, _, tail in graph]
    if _UNDIRECTED in modifiers:
        forward = len(graph)
        graph = graph + [(tail, relation, head) for head, relation, tail in graph]
        fact_rows = {fact: [row, row + forward] for fact, (row,) in fact_rows.items()}
    return graph, fact_rows


def graph_triples(name: str, triples: Mapping[str, Any], seed: int) -> list[IdTriple]:
    """Edges of one graph variant."""
    return graph_variant(name, triples, seed)[0]


def _train_queries(
    queries: Sequence[Mapping[str, Any]],
    query_table: EmbeddingTable,
    column_of: Mapping[int, int],
    fact_rows: Mapping[IdTriple, Sequence[int]],
) -> KgQuerySet:
    """Train queries with the rows of the graph triples that state each query's own fact."""
    selected = sorted((q for q in queries if q["split"] == Split.TRAIN.value), key=lambda q: q["query_id"])
    own_triples = []
    for query in selected:
        rows: list[int] = []
        for answer in query["relevant_ids"] if query["anchor_id"] is not None else ():
            head, tail = (query["anchor_id"], answer) if query["direction"] == "tail" else (answer, query["anchor_id"])
            # Absent in the shuffled and empty graphs: there the fact is not an edge.
            rows.extend(fact_rows.get((head, query["relation_id"], tail), ()))
        own_triples.append(rows)
    return KgQuerySet(
        query_vectors=torch.from_numpy(query_table.select([q["query_id"] for q in selected])),
        relevant=[[column_of[i] for i in q["relevant_ids"]] for q in selected],
        known=[[column_of[i] for i in q["known_ids"] if i in column_of] for q in selected],
        own_triples=own_triples,
    )


def _summary(per_seed: Sequence[Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """Mean and std over seeds of each metric, per report segment."""
    return {
        name: {"answers": per_seed[0][name]["answers"],
               **aggregate_runs([{m: seed[name][m] for m in _METRICS} for seed in per_seed])}
        for name in per_seed[0]
    }


def run_graphsage(config: MkgwRetrievalConfig, data: QueryConfig, graph_name: str, reuse_vectors: bool,
                  grids: Mapping[Gallery, GalleryGrid]) -> None:
    """Train one encoder per seed on one graph variant; score it alone and fused with CLIP per gallery.

    With `reuse_vectors` the entity vectors of an earlier run are scored again instead of
    training, e.g. for another gallery or after a report segment was added.
    """
    if config.training.device.value != "cpu":
        raise ConfigError("the encoder is trained on cpu so that runs match across machines")
    entities, relations, queries, place_splits = _load_splits(config, data)
    embeddings_dir = config.model_embeddings_dir()
    query_table = load_table(embeddings_dir, "queries")
    out_dir = config.runs_dir / "graphsage" / graph_name
    graph, fact_rows = graph_variant(graph_name, read_json(data.out_dir / "triples.json"), config.path_shuffle_seed)

    vectors: dict[int, np.ndarray] = {}
    trained: dict[int, dict[str, Any]] = {}
    if reuse_vectors:
        place_report = out_dir / report_name(Gallery.PLACE)
        previous = read_json(place_report)["seeds"] if place_report.is_file() else {}
        for seed in config.training.seeds:
            vectors[seed] = np.load(out_dir / f"seed-{seed}" / "entity_vectors.npy")
            trained[seed] = {key: previous[str(seed)][key] for key in ("best_epoch", "stop_reason")
                             if str(seed) in previous}
    else:
        entity_features = torch.from_numpy(load_table(embeddings_dir, "entities").select(list(range(len(entities)))))
        column_of = gallery_columns(entities)
        places = torch.tensor(list(column_of))
        triples = torch.tensor(graph, dtype=torch.long).reshape(-1, 3)
        train = _train_queries(queries, query_table, column_of, fact_rows)
        val_data = place_splits[Split.VAL]
        val = KgQuerySet(torch.from_numpy(query_table.select([q["query_id"] for q in val_data.queries])),
                         val_data.relevant, val_data.known, [[] for _ in val_data.queries])
        log.info("graph=%s triples=%d train queries=%d (own fact removed for %d) places=%d", graph_name, len(graph),
                 len(train.relevant), sum(bool(rows) for rows in train.own_triples), len(places))
        for seed in config.training.seeds:
            seed_everything(seed)  # weights are drawn when the model is built
            model = KgEntityEncoder(config.graphsage, entity_features, len(relations))
            result = fit(model, triples, places, train, val, config.training, seed, config.recall_ks)
            seed_dir = out_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_dir / "entity_vectors.npy", result.entity_vectors)
            write_json(seed_dir / "history.json", result.history)
            vectors[seed] = result.entity_vectors
            trained[seed] = {"best_epoch": result.best_epoch, "stop_reason": result.stop_reason.value}
            log.info("seed %d best epoch %d", seed, result.best_epoch)

    for gallery, grid in grids.items():
        splits = place_splits if gallery is Gallery.PLACE else _load_splits(config, data, gallery)[3]
        columns = _column_entities(splits[Split.TEST], entities)
        if max(grid.top_ks) > len(columns):
            raise ConfigError(f"top_k_grid exceeds the {gallery.value} gallery of {len(columns)}")
        seeds: dict[str, Any] = {}
        reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
        graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
        for seed in config.training.seeds:
            column_vectors = vectors[seed][columns]
            scores = {s: query_table.select([q["query_id"] for q in d.queries]) @ column_vectors.T
                      for s, d in splits.items()}
            selected = select_alpha_top_k(splits[Split.VAL], scores[Split.VAL], config.alpha_grid, grid.top_ks)
            seeds[str(seed)] = {**trained[seed], "selected": selected}
            log.info("%s seed %d selected on val: %s", gallery.value, seed, selected)
            for split, split_data in splits.items():
                fused = rerank_scores(split_data.clip_scores, scores[split], selected["alpha"],
                                      int(selected["top_k"]))
                reranked[split].append(_segments(split_data, fused, relations, data, config.recall_ks))
                graph_only[split].append(_segments(split_data, scores[split], relations, data, config.recall_ks))

        report: dict[str, Any] = {
            "graph": graph_name,
            "gallery": gallery.value,
            "gallery_size": len(columns),
            "model_name": config.clip.model_name,
            "graph_triples": len(graph),
            "seeds": seeds,
            "splits": {s.value: {"reranked": _summary(reranked[s]), "graph_only": _summary(graph_only[s])}
                       for s in splits},
        }
        write_json(out_dir / report_name(gallery), report)
        for split in splits:
            for name, row in report["splits"][split.value]["reranked"].items():
                if name.startswith(_LOGGED_PREFIXES):
                    alone = report["splits"][split.value]["graph_only"][name]
                    log.info("%-5s %-5s %-26s n=%4d rerank R@1 %.3f±%.3f R@10 %.3f MRR %.3f | graph only R@1 %.3f "
                             "R@10 %.3f", gallery.value, split.value, name, row["answers"], row["recall@1"]["mean"],
                             row["recall@1"]["std"], row["recall@10"]["mean"], row["mrr"]["mean"],
                             alone["recall@1"]["mean"], alone["recall@10"]["mean"])
        log.info("wrote %s", out_dir / report_name(gallery))


def _entity_image_vectors(entities: Sequence[Mapping[str, Any]], images: EmbeddingTable) -> torch.Tensor:
    """Mean CLIP image vector of every entity; a zero row for an entity without images."""
    vectors = np.zeros((len(entities), images.vectors.shape[1]), dtype=np.float32)
    for row, entity in enumerate(entities):
        if entity["image_files"]:
            vectors[row] = images.select(entity["image_files"]).mean(axis=0)
    return torch.from_numpy(vectors)


def _link_scores(model: KgAdamfMat, data: EntitySplit, gallery: np.ndarray, batch_size: int) -> np.ndarray:
    """(queries x gallery) scores, each anchored row scaled to [0, 1]; zero rows for queries without an anchor."""
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


def run_adamf(config: MkgwRetrievalConfig, data: QueryConfig, model_config: KgAdamfMatConfig,
              training: KgAdamfTrainingConfig, graph_name: str) -> None:
    """Train one AdaMF-MAT model per seed on one graph variant; score link prediction and retrieval."""
    entities, relations, _, splits = _load_splits(config, data)
    embeddings_dir = config.model_embeddings_dir()
    text_vectors = torch.from_numpy(load_table(embeddings_dir, "entities").select(list(range(len(entities)))))
    image_vectors = _entity_image_vectors(entities, load_table(embeddings_dir, "images"))
    gallery = np.array(list(gallery_columns(entities)))
    if max(config.top_k_grid) > len(gallery):
        raise ConfigError(f"top_k_grid exceeds the gallery of {len(gallery)} entities")

    all_triples = read_json(data.out_dir / "triples.json")
    graph = graph_triples(graph_name, all_triples, config.path_shuffle_seed)
    if not graph:
        raise ConfigError("a link-prediction model cannot train without triples")
    val, test = all_triples[Split.VAL.value], all_triples[Split.TEST.value]
    # Every true triple; filters the ranks of other correct answers.
    known = [*graph_triples(graph_name.removesuffix(_SHUFFLED), all_triples, config.path_shuffle_seed), *val, *test]
    completions = known_completions(known)
    out_dir = config.runs_dir / "adamf_mat" / model_config.run_name() / graph_name
    log.info("model=%s graph=%s triples=%d val=%d test=%d gallery=%d", model_config.run_name(), graph_name,
             len(graph), len(val), len(test), len(gallery))

    seeds: dict[str, Any] = {}
    link: list[dict[str, float]] = []
    reranked: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    graph_only: dict[Split, list[dict[str, dict[str, float]]]] = {s: [] for s in splits}
    for seed in config.training.seeds:
        seed_everything(seed)
        model = KgAdamfMat(model_config, len(entities), len(relations), image_vectors, text_vectors)
        result = fit_kg_adamf(model, graph, val, known, training, seed)
        model.load_state_dict(result.state_dict)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_json(seed_dir / "history.json", result.history)
        link.append(link_prediction_metrics(score_fn(model), test, completions, training.eval_batch_size))
        scores = {s: _link_scores(model, d, gallery, training.eval_batch_size) for s, d in splits.items()}
        selected = select_alpha_top_k(splits[Split.VAL], scores[Split.VAL], config.alpha_grid, config.top_k_grid)
        seeds[str(seed)] = {"best_epoch": result.best_epoch, "stop_reason": result.stop_reason.value,
                            "selected": selected}
        log.info("seed %d best epoch %d, test link prediction %s, selected on val: %s", seed, result.best_epoch,
                 {k: round(v, 4) for k, v in link[-1].items()}, selected)
        for split, split_data in splits.items():
            fused = rerank_scores(split_data.clip_scores, scores[split], selected["alpha"], int(selected["top_k"]))
            reranked[split].append(_segments(split_data, fused, relations, data, config.recall_ks))
            graph_only[split].append(_segments(split_data, scores[split], relations, data, config.recall_ks))

    report: dict[str, Any] = {
        "model": model_config.run_name(),
        "graph": graph_name,
        "model_name": config.clip.model_name,
        "graph_triples": len(graph),
        "seeds": seeds,
        "link_prediction": aggregate_runs(link),
        "splits": {s.value: {"reranked": _summary(reranked[s]), "graph_only": _summary(graph_only[s])}
                   for s in splits},
    }
    write_json(out_dir / "benchmark.json", report)
    log.info("test link prediction over seeds: %s",
             {k: round(v["mean"], 4) for k, v in report["link_prediction"].items()})
    for split in splits:
        for name, row in report["splits"][split.value]["reranked"].items():
            if name.startswith(_LOGGED_PREFIXES):
                alone = report["splits"][split.value]["graph_only"][name]
                log.info("%-5s %-26s n=%4d rerank R@1 %.3f±%.3f R@10 %.3f MRR %.3f | graph only R@1 %.3f R@10 %.3f",
                         split.value, name, row["answers"], row["recall@1"]["mean"], row["recall@1"]["std"],
                         row["recall@10"]["mean"], row["mrr"]["mean"],
                         alone["recall@1"]["mean"], alone["recall@10"]["mean"])
    log.info("wrote %s", out_dir / "benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True, choices=["embed", "clip", "graphsage", "adamf"])
    parser.add_argument("--graphs", nargs="+", default=[_CLAIMS],
                        help="graph variants of the graphsage and adamf stages")
    parser.add_argument("--structure-only", action="store_true",
                        help="adamf stage: no image or text modality and no MAT, i.e. plain RotatE")
    parser.add_argument("--gallery", nargs="+", default=[g.value for g in Gallery],
                        choices=[g.value for g in Gallery], help="galleries of the clip and graphsage stages")
    parser.add_argument("--image-dir", help="override paths.image_dir of the query config")
    parser.add_argument("--reuse-vectors", action="store_true",
                        help="graphsage stage: score the saved entity vectors again instead of training")
    parser.add_argument("--split-seed", type=int, help="use the task built with the same --split-seed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = MkgwRetrievalConfig.from_yaml(args.config)
        data = QueryConfig.from_yaml(config.data_config)
        if args.split_seed is not None:
            data = data.with_split_seed(args.split_seed)
            name = f"split-{args.split_seed}"
            config = dataclasses.replace(
                config, embeddings_dir=config.embeddings_dir / name, runs_dir=config.runs_dir / name
            )
        raw = read_mapping(args.config)
        grids = gallery_grids(config, raw, [Gallery(name) for name in args.gallery])
        if args.stage == "embed":
            embed(config, data, Path(args.image_dir) if args.image_dir else data.image_dir)
        elif args.stage == "clip":
            for gallery, grid in grids.items():
                run_clip(config, data, gallery, grid)
        elif args.stage == "graphsage":
            for graph_name in args.graphs:
                run_graphsage(config, data, graph_name, args.reuse_vectors, grids)
        else:
            model_config, adamf_training = load_adamf_configs(raw)
            if args.structure_only:
                model_config = dataclasses.replace(model_config, mat=False, modalities=("structure",))
            for graph_name in args.graphs:
                run_adamf(config, data, model_config, adamf_training, graph_name)
    except AppError:
        log.exception("%s failed", args.stage)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
