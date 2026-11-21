#!/usr/bin/env python3
"""Build the MKG-W caption subset: MKG-W entities in the VG ∩ COCO layout.

Run from the repository root after `build_mkg_w --stage all`:
    python -m scripts.build_mkg_w_captions

Writes to <processed_dir>: images.json, queries.json, graph/{image_graphs,concepts,predicates}.json
and images/<entity id>.jpg (each item's first image, at most 512 pixels a side), plus the split file and a card with the
sizes. After that the VG ∩ COCO scripts run on it with `--config configs/retrieval/mkg_w_captions.yaml`.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from typing import Any

import yaml
from PIL import Image

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.download import image_path
from src.data.json_io import read_json, write_json
from src.data.mkg_w_captions import (
    CaptionSubsetConfig,
    build_vocabulary,
    item_graph,
    item_triples,
    select_items,
)
from src.data.mkg_w_config import MkgwConfig
from src.data.mkg_w_queries import split_entities
from src.data.mkg_w_raw import Triple, display_name
from src.data.schemas import Split

log = logging.getLogger("build_mkg_w_captions")

DEFAULT_CONFIG = "configs/data/mkg_w_captions.yaml"
_JPEG_QUALITY = 90
# CLIP reads 224-pixel crops; a larger copy only costs disk.
_MAX_IMAGE_SIDE = 512


def run(config_path: str) -> None:
    """Select the items, split them, and write every file the retrieval scripts read."""
    data = DataConfig.from_yaml(config_path)
    subset = CaptionSubsetConfig.from_yaml(config_path)
    source = MkgwConfig.from_yaml(subset.source_config).paths.processed_dir
    entities = read_json(source / "entities.json")
    relations = read_json(source / "relations.json")
    triples = [Triple(*row) for rows in read_json(source / "triples.json").values() for row in rows]
    names = [entity["name"] for entity in entities]

    items = select_items(entities, triples, subset)
    if not items:
        raise DataError("no entity passes the subset rules")
    item_split = {item: split for split, ids in split_entities(list(items), subset.fractions, data.split.seed).items()
                  for item in ids}
    touching = item_triples(triples)
    concepts, predicates, concept_of, predicate_of = build_vocabulary(
        item_split, touching, names, [r["label"] or r["pid"] for r in relations], subset.mask_text)
    graphs = [item_graph(item, item_split[item], touching[item], concept_of, predicate_of) for item in sorted(items)]

    out = data.paths.processed_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for item in sorted(items):
        source_file = entities[item]["image_files"][0]
        with Image.open(source / "images" / source_file) as picture:
            copy = picture.convert("RGB")
            copy.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
            copy.save(image_path(out / "images", item), "JPEG", quality=_JPEG_QUALITY)
        split = item_split[item].value
        captions = [{"caption_id": f"cap-{item}-{n}", "text": text} for n, text in enumerate(items[item])]
        images.append({"vg_image_id": item, "name": display_name(names[item]), "source_image": source_file,
                       "split": split, "captions": captions})
        queries.extend({"query_id": c["caption_id"], "text": c["text"], "relevant_ids": [item], "split": split}
                       for c in captions)

    write_json(out / "images.json", images)
    write_json(out / "queries.json", queries)
    write_json(out / "graph" / "image_graphs.json", graphs)
    write_json(out / "graph" / "concepts.json", concepts)
    write_json(out / "graph" / "predicates.json", predicates)
    by_split = {s.value: sorted(i for i, v in item_split.items() if v is s) for s in Split}
    write_json(data.paths.split_file, {"config": {"min_triples": subset.min_triples, "seed": data.split.seed,
                                                  "fractions": {s.value: f for s, f in subset.fractions.items()}},
                                       **by_split})
    triple_counts = [len(g["triplets"]) for g in graphs]
    card = {
        "items": {s: len(ids) for s, ids in by_split.items()},
        "captions": dict(sorted(Counter(q["split"] for q in queries).items())),
        "captions_per_item": subset.captions_per_item,
        "concepts": len(concepts),
        "concepts_seen_in_train": sum(c["train_images"] > 0 for c in concepts),
        "predicates": len(predicates),
        "triples_per_item": {"min": min(triple_counts), "median": statistics.median(triple_counts),
                             "max": max(triple_counts)},
    }
    data.paths.graph_card_file.write_text(
        "# MKG-W caption subset card. Written by `python -m scripts.build_mkg_w_captions`; do not edit by hand.\n"
        + yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
    log.info("card: %s", card)
    log.info("wrote %s", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(args.config)
    except AppError:
        log.exception("build failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
