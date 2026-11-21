#!/usr/bin/env python3
"""Build the VG ∩ COCO variant whose image graphs carry WordNet IsA edges, and its indirect queries.

    python -m scripts.build_kg_variant

Reads the base dataset of configs/retrieval/vg_coco.yaml and writes, for
configs/retrieval/vg_coco_kg.yaml:
- <kg processed>/images.json, queries.json: copies of the base files;
- <kg processed>/graph/{concepts,predicates,image_graphs}.json: hypernyms (up to DEPTH
  levels) as concepts, `is a` as a predicate, and the IsA triplets in every image graph;
- <kg processed>/graph/indirect_queries.json: indirect queries and their direct twins;
- <kg embeddings>/<model>/: the base image and caption tables (hard links), CLIP vectors of
  the extended concepts and predicates, the indirect queries, and meta.json.
Needs nltk with the WordNet corpus.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.kg_edges import ISA_PREDICATE, add_isa_edges, build_indirect_queries, extend_vocabulary
from src.data.sources import sha256_of
from src.data.synsets import synset_lemma
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.store import META_FILE, EmbeddingTable, encode_in_batches, save_table, table_paths
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("build_kg_variant")

BASE_CONFIG = "configs/retrieval/vg_coco.yaml"
KG_CONFIG = "configs/retrieval/vg_coco_kg.yaml"
DEPTH = 2
QUERY_SEED = 13
# Hypernyms nearer the WordNet root than this (entity, abstraction, object...) never name a query.
MIN_QUERY_DEPTH = 4


def hypernym_chain(synset: str, depth: int) -> list[str]:
    """Hypernym chain up to `depth` levels, the alphabetically first parent at each; empty if unknown."""
    try:
        current = wn.synset(synset)
    except (WordNetError, ValueError):
        return []
    chain = []
    for _ in range(depth):
        # NLTK lists the parents of a multi-parent synset in no fixed order; sort for reproducibility.
        parents = sorted(current.hypernyms() or current.instance_hypernyms(), key=lambda p: p.name())
        if not parents:
            break
        current = parents[0]
        chain.append(current.name())
    return chain


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    """Build the KG variant end to end."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=BASE_CONFIG)
    parser.add_argument("--kg", default=KG_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        base, kg = RetrievalConfig.from_yaml(args.base), RetrievalConfig.from_yaml(args.kg)
        base_dir = DataConfig.from_yaml(base.data_config).paths.processed_dir
        kg_dir = DataConfig.from_yaml(kg.data_config).paths.processed_dir
        base_concepts = read_json(base_dir / "graph" / "concepts.json")
        concepts, predicates, chains = extend_vocabulary(
            base_concepts,
            read_json(base_dir / "graph" / "predicates.json"),
            hypernym_chain, DEPTH, synset_lemma,
        )
        isa_id = next(int(p["predicate_id"]) for p in predicates if p["text"] == ISA_PREDICATE)
        graphs = read_json(base_dir / "graph" / "image_graphs.json")
        text = {int(c["concept_id"]): c["text"] for c in concepts}
        predicate_text = {int(p["predicate_id"]): p["text"] for p in predicates}
        # Queries name only specific hypernyms: "abstraction on table" would test nothing.
        synset_of = {int(c["concept_id"]): str(c["synset"]) for c in concepts if c.get("synset")}
        query_chains = {
            child: [h for h in parents if wn.synset(synset_of[h]).min_depth() >= MIN_QUERY_DEPTH]
            for child, parents in chains.items()
        }
        queries = build_indirect_queries(graphs, query_chains, text, predicate_text, QUERY_SEED)
        kg_graphs = [add_isa_edges(g, chains, isa_id) for g in graphs]

        for name in ("images.json", "queries.json"):
            _link_or_copy(base_dir / name, kg_dir / name)
        write_json(kg_dir / "graph" / "concepts.json", concepts)
        write_json(kg_dir / "graph" / "predicates.json", predicates)
        write_json(kg_dir / "graph" / "image_graphs.json", kg_graphs, indent=None)
        write_json(kg_dir / "graph" / "indirect_queries.json", queries)
        added = sum(len(g["triplets"]) for g in kg_graphs) - sum(len(g["triplets"]) for g in graphs)
        log.info("concepts %d (+%d hypernyms), IsA triplets %d, indirect queries %d; top hypernyms %s",
                 len(concepts), len(concepts) - len(base_concepts), added,
                 sum(1 for q in queries if q["kind"] == "indirect"),
                 Counter(text[q["hypernym"]] for q in queries if q["kind"] == "indirect").most_common(8))

        base_emb, kg_emb = base.model_embeddings_dir(), kg.model_embeddings_dir()
        for name in ("images", "queries"):
            for path in table_paths(base_emb, name):
                _link_or_copy(path, kg_emb / path.name)
        encoder = ClipEncoder(kg.clip.model_name, kg.clip.device)
        batch = kg.clip.text_batch_size
        tables = {
            "concepts": EmbeddingTable(
                ids=tuple(int(c["concept_id"]) for c in concepts),
                vectors=encode_in_batches([c["text"] for c in concepts], encoder.encode_texts, batch),
            ),
            "predicates": EmbeddingTable(
                ids=tuple(int(p["predicate_id"]) for p in predicates),
                vectors=encode_in_batches([p["text"] for p in predicates], encoder.encode_texts, batch),
            ),
            "indirect_queries": EmbeddingTable(
                ids=tuple(q["query_id"] for q in queries),
                vectors=encode_in_batches([q["text"] for q in queries], encoder.encode_texts, batch),
            ),
        }
        for name, table in tables.items():
            save_table(kg_emb, name, table)
        inputs = [kg_dir / "images.json", kg_dir / "queries.json",
                  kg_dir / "graph" / "concepts.json", kg_dir / "graph" / "predicates.json"]
        write_json(kg_emb / META_FILE, {
            "model_name": kg.clip.model_name,
            "base_embeddings": str(base_emb),
            "hypernym_depth": DEPTH,
            "tables": {
                name: {"rows": rows, "sha256": {p.name: sha256_of(p) for p in table_paths(kg_emb, name)}}
                for name, rows in (
                    ("images", None), ("queries", None), ("concepts", len(concepts)), ("predicates", len(predicates)),
                )
            },
            "inputs_sha256": {str(p): sha256_of(p) for p in inputs},
        })
        log.info("wrote %s and %s", kg_dir, kg_emb)
    except AppError:
        log.exception("building the KG variant failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
