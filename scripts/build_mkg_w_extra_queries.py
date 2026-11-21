#!/usr/bin/env python3
"""Build the extra MKG-W test sets that are not one triple turned into a sentence.

Run from the repository root after `build_mkg_w --stage build`:
    python -m scripts.build_mkg_w_extra_queries

Writes <processed_dir>/extra_queries.json with two lists of query records:
- `chain`: generated queries that need two facts in a row; the entity between them is not named.
- `handwritten`: the free-text queries of the hand-written file, checked against the triples.
Both are test-only. Each record lists the entities its text names in `anchor_ids`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from src.core.config_parsing import read_mapping
from src.core.exceptions import AppError, DataError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.mkg_w_handwritten import resolve_handwritten
from src.data.mkg_w_queries import EXTRA_QUERIES_FILE, build_chain_queries, direction_stats
from src.data.mkg_w_raw import Triple, display_name
from src.data.schemas import Split

log = logging.getLogger("build_mkg_w_extra_queries")

DEFAULT_CONFIG = "configs/data/mkg_w.yaml"
DEFAULT_HANDWRITTEN = "data/splits/mkg_w_handwritten_queries.yaml"


def run(config: MkgwConfig, handwritten_file: Path) -> None:
    """Build both sets from the processed dataset and write them."""
    processed = config.paths.processed_dir
    entities = read_json(processed / "entities.json")
    relations = read_json(processed / "relations.json")
    triples = {split: [Triple(*row) for row in rows]
               for split, rows in ((Split(name), rows) for name, rows in read_json(processed / "triples.json").items())}
    names = [entity["name"] for entity in entities]
    has_image = [bool(entity["image_files"]) for entity in entities]
    relation_of = {relation["pid"]: relation["relation_id"] for relation in relations}
    templates = {relation_of[pid]: template for pid, template in config.templates.items()}

    stats = direction_stats(triples[Split.TRAIN], config.rules)
    chain = [{**query.to_dict(), "anchor_ids": [query.anchor_id]} for query in build_chain_queries(
        triples, Split.TEST, stats, templates, names, has_image, config.rules.max_answers)]

    items = read_mapping(handwritten_file).get("queries")
    if not isinstance(items, list) or not items:
        raise DataError(f"{handwritten_file} has no `queries` list")
    entity_of = {display_name(name): entity_id for entity_id, name in enumerate(names)}
    handwritten = resolve_handwritten(items, entity_of, relation_of, triples, has_image)

    write_json(processed / EXTRA_QUERIES_FILE, {"chain": chain, "handwritten": handwritten})
    log.info("chain: %d queries, %d answers", len(chain), sum(len(q["relevant_ids"]) for q in chain))
    log.info("handwritten: %d queries %s", len(handwritten), dict(Counter(q["kind"] for q in handwritten)))
    log.info("wrote %s", processed / EXTRA_QUERIES_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--handwritten", default=DEFAULT_HANDWRITTEN, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(MkgwConfig.from_yaml(args.config), args.handwritten)
    except AppError:
        log.exception("build failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
