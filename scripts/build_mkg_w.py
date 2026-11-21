#!/usr/bin/env python3
"""Build the MKG-W retrieval dataset stage by stage.

Run from the repository root:
    python -m scripts.build_mkg_w --stage sources
    python -m scripts.build_mkg_w --stage images
    python -m scripts.build_mkg_w --stage build
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from src.core.exceptions import AppError, ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.mkg_w_images import extract_images, image_members
from src.data.mkg_w_queries import (
    Direction,
    DirectionStat,
    EntityQuery,
    anchor_answer_hops,
    build_direct_queries,
    build_indirect_queries,
    direction_stats,
    split_entities,
)
from src.data.mkg_w_raw import (
    Triple,
    display_name,
    match_folders,
    read_descriptions,
    read_entity_names,
    read_id_file,
    read_triples,
)
from src.data.schemas import Split
from src.data.sources import ensure_source, load_manifest, save_manifest, sha256_of
from src.data.yaml_io import write_yaml

log = logging.getLogger("build_mkg_w")

DEFAULT_CONFIG = "configs/data/mkg_w.yaml"
_TRIPLE_SOURCES = {Split.TRAIN: "mkg_w_train2id", Split.VAL: "mkg_w_valid2id", Split.TEST: "mkg_w_test2id"}
_CARD_HEADER = (
    "MKG-W retrieval dataset card. Written by `python -m scripts.build_mkg_w --stage build`.\n"
    "Do not edit by hand; change configs/data/mkg_w.yaml and rebuild."
)


def run_sources(config: MkgwConfig) -> None:
    """Fetch missing raw files and verify every file against the manifest."""
    manifest = load_manifest(config.paths.manifest_file)
    for spec in config.sources:
        manifest[spec.name] = ensure_source(spec, config.paths.raw_dir, manifest)
    save_manifest(config.paths.manifest_file, manifest)
    log.info("manifest: %s (%d files)", config.paths.manifest_file, len(manifest))


def _entity_names(config: MkgwConfig) -> tuple[list[str], list[str]]:
    qids = read_id_file(config.raw_path("mkg_w_entity2id"))
    return qids, read_entity_names(config.raw_path("mkg_w_ent_links"), qids)


def run_images(config: MkgwConfig) -> None:
    """Match archive folders to entities, extract the images that decode, write the image index."""
    _, names = _entity_names(config)
    out_dir = config.paths.processed_dir
    with zipfile.ZipFile(config.raw_path("mkg_w_images")) as archive:
        members = image_members(archive)
        folder_of_entity = match_folders(dict(enumerate(names)), members)
        log.info("archive: %d folders, %d matched to entities", len(members), len(folder_of_entity))
        files, unreadable = extract_images(archive, folder_of_entity, members, out_dir / "images")
    write_json(out_dir / "image_index.json", {
        "archive_folders": len(members),
        "matched_folders": len(folder_of_entity),
        "entities_with_images": len(files),
        "images": sum(len(v) for v in files.values()),
        "unreadable": unreadable,
        "files": {str(entity_id): paths for entity_id, paths in files.items()},
    }, indent=None)
    log.info("kept %d images of %d entities; %d files did not decode",
             sum(len(v) for v in files.values()), len(files), len(unreadable))


def _direction_rows(stats: dict[tuple[int, Direction], DirectionStat], pids: list[str],
                    labels: dict[int, str], queries: list[EntityQuery]) -> list[dict[str, Any]]:
    per_direction: Counter[tuple[int, Direction, Split]] = Counter()
    for query in queries:
        if query.relation_id is not None and query.direction is not None:
            per_direction[(query.relation_id, query.direction, query.split)] += 1
    rows = []
    for (relation_id, direction), stat in sorted(stats.items(), key=lambda kv: (-kv[1].train_triples, kv[0][1].value)):
        rows.append({
            "relation": pids[relation_id],
            "label": labels.get(relation_id, ""),
            "direction": direction.value,
            "train_triples": stat.train_triples,
            "unique_answers": stat.unique_answers,
            "prior_hits": stat.prior_hits,
            "status": stat.status.value,
            "queries": {s.value: per_direction[(relation_id, direction, s)] for s in Split},
        })
    return rows


def run_build(config: MkgwConfig) -> None:
    """Write entities, relations, triples and queries, plus the split file and the dataset card."""
    qids, names = _entity_names(config)
    pids = read_id_file(config.raw_path("mkg_w_relation2id"))
    triples: dict[Split, list[Triple]] = {
        split: read_triples(config.raw_path(source), len(qids), len(pids))
        for split, source in _TRIPLE_SOURCES.items()
    }
    index_path = config.paths.processed_dir / "image_index.json"
    if not index_path.exists():
        raise DataError(f"{index_path} is missing; run `--stage images` first")
    image_files = {int(k): v for k, v in read_json(index_path)["files"].items()}
    has_image = [entity_id in image_files for entity_id in range(len(qids))]
    by_name = read_descriptions(config.raw_path("mkg_w_descriptions"))
    descriptions = {i: by_name[name] for i, name in enumerate(names) if name in by_name}

    unknown = sorted(pid for pid in config.templates if pid not in pids)
    if unknown:
        raise ConfigError(f"templates for relations that are not in the dataset: {unknown}")
    templates = {pids.index(pid): template for pid, template in config.templates.items()}

    stats = direction_stats(triples[Split.TRAIN], config.rules)
    indirect = build_indirect_queries(triples, stats, templates, names, has_image, config.rules.max_answers)
    hops = anchor_answer_hops(indirect, triples[Split.TRAIN], config.rules.max_hops)
    indirect = [replace(query, hops=h) for query, h in zip(indirect, hops, strict=True)]
    entity_split = split_entities(sorted(image_files), config.direct_split.fractions, config.direct_split.seed)
    direct = build_direct_queries(entity_split, names, {i: d for i, d in descriptions.items() if has_image[i]})
    queries = direct + indirect
    if len({q.query_id for q in queries}) != len(queries):
        raise DataError("duplicate query ids")

    out_dir = config.paths.processed_dir
    write_json(out_dir / "entities.json", [
        {"entity_id": i, "qid": qids[i], "name": display_name(names[i]),
         "description": descriptions.get(i), "image_files": image_files.get(i, [])}
        for i in range(len(qids))
    ], indent=None)
    write_json(out_dir / "relations.json", [
        {"relation_id": i, "pid": pid, "label": templates[i].label if i in templates else None}
        for i, pid in enumerate(pids)
    ])
    write_json(out_dir / "triples.json", {
        split.value: [[t.head, t.relation, t.tail] for t in rows] for split, rows in triples.items()
    }, indent=None)
    write_json(out_dir / "queries.json", [q.to_dict() for q in queries], indent=None)

    counts: Counter[tuple[str, str, str]] = Counter((q.group.value, q.kind.value, q.split.value) for q in queries)
    query_counts = {
        f"{group}/{kind}": {s.value: counts[(group, kind, s.value)] for s in Split}
        for group, kind in sorted({(g, k) for g, k, _ in counts})
    }
    outputs = {name: sha256_of(out_dir / name)
               for name in ("entities.json", "relations.json", "triples.json", "queries.json")}
    write_json(config.paths.split_file, {
        "config": {"query_rules": config.rules.to_dict(), "direct_split": config.direct_split.to_dict()},
        "inputs_sha256": {spec.name: sha256_of(config.paths.raw_dir / spec.target)
                          for spec in config.sources if spec.name != "mkg_w_images"},
        "outputs_sha256": outputs,
        "query_counts": query_counts,
        "direct_entities": {split.value: ids for split, ids in entity_split.items()},
    }, indent=None)
    answer_sizes = Counter(len(q.relevant_ids) + len(q.known_ids) for q in indirect)
    hop_counts: Counter[tuple[str, Split]] = Counter(
        (str(q.hops) if q.hops else f"over_{config.rules.max_hops}_or_none", q.split) for q in indirect
    )
    write_yaml(config.paths.card_file, {
        "entities": len(qids),
        "entities_with_images": len(image_files),
        "images": sum(len(v) for v in image_files.values()),
        "entities_with_description": len(descriptions),
        "relations": len(pids),
        "triples": {split.value: len(rows) for split, rows in triples.items()},
        "query_rules": config.rules.to_dict(),
        "query_counts": query_counts,
        "indirect_queries_by_answer_count": {str(k): answer_sizes[k] for k in sorted(answer_sizes)},
        "indirect_queries_by_hops": {
            key: {s.value: hop_counts[(key, s)] for s in Split} for key in sorted({k for k, _ in hop_counts})
        },
        "directions": _direction_rows(stats, pids, {i: t.label for i, t in templates.items()}, indirect),
    }, header=_CARD_HEADER)
    for key, row in query_counts.items():
        log.info("%-22s %s", key, row)
    log.info("outputs sha256: %s", outputs)


STAGES: dict[str, Callable[[MkgwConfig], None]] = {
    "sources": run_sources,
    "images": run_images,
    "build": run_build,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        config = MkgwConfig.from_yaml(args.config)
        for name in list(STAGES) if args.stage == "all" else [args.stage]:
            log.info("stage %s", name)
            STAGES[name](config)
    except AppError:
        log.exception("build failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
