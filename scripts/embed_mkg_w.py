#!/usr/bin/env python3
"""Compute and store frozen CLIP embeddings for MKG-W.

Run from the repository root after `build_mkg_w --stage build`:
    python -m scripts.embed_mkg_w

Writes four tables to <embeddings_dir>/<model name>/: `images` (one row per image file),
`queries` (query texts), `entities` (the name of every entity, as text) and `descriptions`
(the description of every entity; its name when it has none), plus meta.json.

`--tables descriptions` recomputes only the named tables and keeps the others as they are.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.core.exceptions import AppError, ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.mkg_w_config import MkgwConfig
from src.data.sources import sha256_of
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import (
    META_FILE,
    EmbeddingTable,
    encode_in_batches,
    save_table,
    table_paths,
    verify_inputs_unchanged,
)
from src.retrieval.mkg_w_config import MkgwRetrievalConfig

log = logging.getLogger("embed_mkg_w")

DEFAULT_CONFIG = "configs/retrieval/mkg_w.yaml"
_LOG_EVERY_BATCHES = 40

Encode = Callable[[Sequence[Any]], np.ndarray]


def _with_progress(name: str, total: int, encode: Encode) -> Encode:
    """Wrap an encoder so it logs how many items are done every few batches."""
    state = {"done": 0, "batches": 0}

    def wrapped(batch: Sequence[Any]) -> np.ndarray:
        vectors = encode(batch)
        state["done"] += len(batch)
        state["batches"] += 1
        if state["batches"] % _LOG_EVERY_BATCHES == 0 or state["done"] == total:
            log.info("%s: %d/%d", name, state["done"], total)
        return vectors

    return wrapped


def run(config: MkgwRetrievalConfig, only: Sequence[str] = ()) -> None:
    """Encode image files, query texts, entity names and descriptions; write tables and meta.

    `only` limits the work to those tables; the meta entries of the other tables are kept.
    """
    processed = MkgwConfig.from_yaml(config.data_config).paths.processed_dir
    inputs = {"entities": processed / "entities.json", "queries": processed / "queries.json"}
    entities = read_json(inputs["entities"])
    queries = read_json(inputs["queries"])

    image_ids = [file for entity in entities for file in entity["image_files"]]
    image_files = [processed / "images" / file for file in image_ids]
    missing = [str(p) for p in image_files if not p.is_file()]
    if missing:
        raise DataError(f"{len(missing)} image files are missing, e.g. {missing[:3]}")

    encoder = ClipEncoder(config.clip.model_name, config.clip.device)
    log.info("model=%s device=%s dim=%d", config.clip.model_name, encoder.device, encoder.dim)
    texts, text_batch = encoder.encode_texts, config.clip.text_batch_size
    jobs: dict[str, tuple[list[Any], list[Any], Encode, int]] = {
        "entities": ([e["entity_id"] for e in entities], [e["name"] for e in entities], texts, text_batch),
        "descriptions": ([e["entity_id"] for e in entities], [e["description"] or e["name"] for e in entities],
                         texts, text_batch),
        "queries": ([q["query_id"] for q in queries], [q["text"] for q in queries], texts, text_batch),
        "images": (image_ids, image_files, encoder.encode_image_files, config.clip.image_batch_size),
    }
    unknown = set(only) - set(jobs)
    if unknown:
        raise ConfigError(f"unknown tables {sorted(unknown)}; choose from {sorted(jobs)}")
    out_dir = config.model_embeddings_dir()
    kept: dict[str, Any] = {}
    if only:
        verify_inputs_unchanged(out_dir)
        kept = read_json(out_dir / META_FILE)["tables"]
        jobs = {name: jobs[name] for name in only}
    tables: dict[str, EmbeddingTable] = {}
    for name, (ids, items, encode, batch_size) in jobs.items():
        vectors = encode_in_batches(items, _with_progress(name, len(items), encode), batch_size)
        tables[name] = EmbeddingTable(ids=tuple(ids), vectors=vectors)
        save_table(out_dir, name, tables[name])

    write_json(out_dir / META_FILE, {
        "model_name": config.clip.model_name,
        "device": encoder.device,
        "dim": encoder.dim,
        "tables": kept | {
            name: {"rows": len(table.ids), "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)}}
            for name, table in tables.items()
        },
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs.values()},
    })
    log.info("tables: %s", {name: table.vectors.shape for name, table in tables.items()})
    log.info("wrote %s", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tables", nargs="+", default=[], help="recompute only these tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(MkgwRetrievalConfig.from_yaml(args.config), args.tables)
    except AppError:
        log.exception("embedding failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
