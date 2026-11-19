#!/usr/bin/env python3
"""How often captions name a triplet of their own image graph, with three ways of matching object names.

    python -m scripts.caption_coverage

- exact: the concept's display text (what training uses for predicate-swap negatives);
- synonyms: also the WordNet lemmas of the concept's synset (`adult male` for man);
- hypernyms: also the lemmas of its hypernyms up to two levels (`tree` for an oak).
A caption names a triplet when it contains the predicate and one accepted name of each
object. Writes <runs_dir>/analysis/caption_coverage.json. Needs nltk with WordNet.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from typing import Any

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.text_match import names, names_any
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("caption_coverage")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
HYPERNYM_DEPTH = 2


def _lemmas(synset: Any) -> set[str]:
    return {name.replace("_", " ").lower() for name in synset.lemma_names()}


def name_sets(concepts: list[dict[str, Any]]) -> dict[str, dict[int, tuple[str, ...]]]:
    """Accepted names of every concept under each matching mode."""
    modes: dict[str, dict[int, tuple[str, ...]]] = defaultdict(dict)
    for concept in concepts:
        cid, text = int(concept["concept_id"]), str(concept["text"])
        synonyms, hypernyms = {text}, {text}
        try:
            synset = wn.synset(concept["synset"]) if concept.get("synset") else None
        except (WordNetError, ValueError):
            synset = None
        if synset is not None:
            synonyms |= _lemmas(synset)
            hypernyms |= synonyms
            frontier = [synset]
            for _ in range(HYPERNYM_DEPTH):
                frontier = [p for s in frontier for p in s.hypernyms()]
                for parent in frontier:
                    hypernyms |= _lemmas(parent)
        modes["exact"][cid] = (text,)
        modes["synonyms"][cid] = tuple(sorted(synonyms))
        modes["hypernyms"][cid] = tuple(sorted(hypernyms))
    return modes


def main() -> int:
    """Count captions naming a triplet of their image, per split and matching mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
        concepts = read_json(processed / "graph" / "concepts.json")
        predicate = {int(p["predicate_id"]): p["text"] for p in read_json(processed / "graph" / "predicates.json")}
        graph_of = {int(g["vg_image_id"]): g for g in read_json(processed / "graph" / "image_graphs.json")}
        modes = name_sets(concepts)
        report: dict[str, Any] = {"hypernym_depth": HYPERNYM_DEPTH, "splits": {}}
        for split in ("train", "test"):
            captions = [q for q in read_json(processed / "queries.json") if q["split"] == split]
            counts = {}
            for mode, accepted in modes.items():
                named = 0
                for caption in captions:
                    graph, text = graph_of[int(caption["relevant_ids"][0])], caption["text"]
                    named += any(
                        int(s) != int(o) and names(text, predicate[int(p)])
                        and names_any(text, accepted[int(s)]) and names_any(text, accepted[int(o)])
                        for s, p, o in graph["triplets"]
                    )
                counts[mode] = {"captions": len(captions), "naming_a_triplet": named,
                                "share": round(named / len(captions), 4)}
                log.info("%s %-9s %d / %d captions name a triplet (%.1f%%)",
                         split, mode, named, len(captions), 100 * named / len(captions))
            report["splits"][split] = counts
        write_json(config.runs_dir / "analysis" / "caption_coverage.json", report)
    except AppError:
        log.exception("caption coverage failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
