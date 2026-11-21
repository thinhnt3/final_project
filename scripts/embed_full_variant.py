#!/usr/bin/env python3
"""CLIP tables of the full VG ∩ COCO variant, reusing every vector the base tables already hold.

    python -m scripts.embed_full_variant

Images and captions present in the base tables keep their exact vectors (so validation
and test are identical to the base variant); only the extra train images and captions are
encoded. Concepts and predicates are the base tables (same vocabulary). Writes the tables
and a meta.json that `verify_inputs_unchanged` accepts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.download import image_path
from src.data.json_io import read_json, write_json
from src.data.sources import sha256_of
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import META_FILE, EmbeddingTable, encode_in_batches, load_table, save_table, table_paths
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("embed_full_variant")


def _extend(base: EmbeddingTable, ids: list[int | str], encode_missing) -> EmbeddingTable:  # type: ignore[no-untyped-def]
    """Table over `ids`: base rows where present, newly encoded rows for the rest."""
    rows = base.row_of()
    missing = [i for i in ids if i not in rows]
    new = encode_missing(missing) if missing else np.zeros((0, base.vectors.shape[1]), dtype=np.float32)
    new_row = {item: r for r, item in enumerate(missing)}
    vectors = np.stack([base.vectors[rows[i]] if i in rows else new[new_row[i]] for i in ids]).astype(np.float32)
    log.info("%d rows: %d reused, %d encoded", len(ids), len(ids) - len(missing), len(missing))
    return EmbeddingTable(ids=tuple(ids), vectors=vectors)


def main() -> int:
    """Encode what is new and write the full-variant tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="configs/retrieval/vg_coco.yaml")
    parser.add_argument("--full", default="configs/retrieval/vg_coco_full.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        base, full = RetrievalConfig.from_yaml(args.base), RetrievalConfig.from_yaml(args.full)
        processed = DataConfig.from_yaml(full.data_config).paths.processed_dir
        base_dir, out_dir = base.model_embeddings_dir(), full.model_embeddings_dir()
        images, queries = read_json(processed / "images.json"), read_json(processed / "queries.json")
        image_ids = [int(r["vg_image_id"]) for r in images]
        missing_files = [i for i in image_ids if not image_path(processed / "images", i).is_file()]
        if missing_files:
            raise DataError(f"{len(missing_files)} image files are missing, e.g. {missing_files[:5]}")
        encoder = ClipEncoder(full.clip.model_name, full.clip.device)
        text_of = {q["query_id"]: q["text"] for q in queries}
        tables = {
            "images": _extend(
                load_table(base_dir, "images"), list(image_ids),
                lambda ids: encode_in_batches(
                    [image_path(processed / "images", i) for i in ids], encoder.encode_image_files,
                    full.clip.image_batch_size,
                ),
            ),
            "queries": _extend(
                load_table(base_dir, "queries"), [q["query_id"] for q in queries],
                lambda ids: encode_in_batches(
                    [text_of[i] for i in ids], encoder.encode_texts, full.clip.text_batch_size
                ),
            ),
            "concepts": load_table(base_dir, "concepts"),
            "predicates": load_table(base_dir, "predicates"),
        }
        for name, table in tables.items():
            save_table(out_dir, name, table)
        inputs = [processed / "images.json", processed / "queries.json",
                  processed / "graph" / "concepts.json", processed / "graph" / "predicates.json"]
        write_json(out_dir / META_FILE, {
            "model_name": full.clip.model_name,
            "device": encoder.device,
            "base_embeddings": str(base_dir),
            "tables": {
                name: {"rows": len(t.ids), "sha256": {p.name: sha256_of(p) for p in table_paths(out_dir, name)}}
                for name, t in tables.items()
            },
            "inputs_sha256": {str(Path(p)): sha256_of(p) for p in inputs},
        })
        log.info("wrote %s", out_dir)
    except AppError:
        log.exception("embedding the full variant failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
