#!/usr/bin/env python3
"""Compute and store frozen CLIP embeddings for the VG ∩ COCO subset.

Run from the repository root after `build_vg_coco --stage graph`:
    python -m scripts.embed_vg_coco

Writes four tables (images, queries, concepts, predicates) and meta.json to
<embeddings_dir>/<model name>/. Later steps read these files and never call CLIP again.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.download import image_path
from src.data.json_io import read_json, write_json
from src.data.sources import sha256_of
from src.embeddings.checks import caption_image_alignment
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import META_FILE, EmbeddingTable, encode_in_batches, save_table, table_paths
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("embed_vg_coco")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
_LOG_EVERY_BATCHES = 20


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


def run(config: RetrievalConfig) -> None:
    """Encode images, captions, concept texts and predicate texts; write tables and meta."""
    data = DataConfig.from_yaml(config.data_config)
    processed = data.paths.processed_dir
    inputs = {
        "images": processed / "images.json",
        "queries": processed / "queries.json",
        "concepts": processed / "graph" / "concepts.json",
        "predicates": processed / "graph" / "predicates.json",
    }
    images = read_json(inputs["images"])
    queries = read_json(inputs["queries"])
    concepts = read_json(inputs["concepts"])
    predicates = read_json(inputs["predicates"])

    image_ids = [int(r["vg_image_id"]) for r in images]
    image_files = [image_path(processed / "images", i) for i in image_ids]
    missing = [str(p) for p in image_files if not p.is_file()]
    if missing:
        raise DataError(f"{len(missing)} image files are missing, e.g. {missing[:3]}")
    # Checked before the model is loaded: a bad reference would otherwise surface only after
    # every table has been encoded and written.
    known_images = set(image_ids)
    orphans = [
        q["query_id"] for q in queries
        if len(q["relevant_ids"]) != 1 or int(q["relevant_ids"][0]) not in known_images
    ]
    if orphans:
        raise DataError(
            f"{len(orphans)} queries do not have exactly one relevant image listed in images.json, e.g. {orphans[:3]}"
        )

    encoder = ClipEncoder(config.clip.model_name, config.clip.device)
    log.info("model=%s device=%s dim=%d", config.clip.model_name, encoder.device, encoder.dim)
    text_batch, image_batch = config.clip.text_batch_size, config.clip.image_batch_size
    texts = encoder.encode_texts
    jobs: dict[str, tuple[list[Any], list[Any], Encode, int]] = {
        "concepts": ([c["concept_id"] for c in concepts], [c["text"] for c in concepts], texts, text_batch),
        "predicates": ([p["predicate_id"] for p in predicates], [p["text"] for p in predicates], texts, text_batch),
        "queries": ([q["query_id"] for q in queries], [q["text"] for q in queries], texts, text_batch),
        "images": (image_ids, image_files, encoder.encode_image_files, image_batch),
    }

    out_dir = config.model_embeddings_dir()
    tables: dict[str, EmbeddingTable] = {}
    for name, (ids, items, encode, batch_size) in jobs.items():
        vectors = encode_in_batches(items, _with_progress(name, len(items), encode), batch_size)
        tables[name] = EmbeddingTable(ids=tuple(ids), vectors=vectors)
        save_table(out_dir, name, tables[name])

    image_rows = tables["images"].row_of()
    alignment = caption_image_alignment(
        tables["queries"].vectors,
        tables["images"].vectors,
        [image_rows[int(q["relevant_ids"][0])] for q in queries],
        seed=data.split.seed,
    )
    meta = {
        "model_name": config.clip.model_name,
        "device": encoder.device,
        "dim": encoder.dim,
        "tables": {
            name: {
                "rows": len(table.ids),
                "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)},
            }
            for name, table in tables.items()
        },
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs.values()},
        "caption_image_alignment": vars(alignment),
    }
    write_json(out_dir / META_FILE, meta)
    log.info("tables: %s", {name: table.vectors.shape for name, table in tables.items()})
    log.info("caption↔own image cosine %.4f, caption↔random image %.4f (%d captions)",
             alignment.own_mean, alignment.random_mean, alignment.pairs)
    log.info("wrote %s", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(RetrievalConfig.from_yaml(args.config))
    except AppError:
        log.exception("embedding failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
