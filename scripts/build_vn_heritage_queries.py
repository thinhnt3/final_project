#!/usr/bin/env python3
"""Build the retrieval task of the VN heritage core: entities, graph variants and queries.

Run from the repository root after `build_vn_heritage_core`:
    python -m scripts.build_vn_heritage_queries

Writes to <out_dir>: entities.json, relations.json, triples.json (train / val / test claims
and the article edges under every naming scheme), queries.json and card.json (counts).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core.exceptions import AppError, DataError
from src.data.json_io import write_json
from src.data.mkg_w_queries import (
    EntityQuery,
    anchor_answer_hops,
    build_direct_queries,
    build_indirect_queries,
    direction_stats,
)
from src.data.schemas import Split
from src.data.vn_heritage_articles import Mention, mention_triples
from src.data.vn_heritage_queries import (
    NamedTriple,
    article_triples,
    density_tier,
    first_label,
    split_claims,
    to_ids,
    triples_within_two_hops,
    worded_directions,
)
from src.data.vn_heritage_query_config import QueryConfig

log = logging.getLogger("build_vn_heritage_queries")

DEFAULT_CONFIG = "configs/data/vn_heritage_queries.yaml"
# The core's triple file also lists image -> place edges; they are not knowledge.
_IMAGE_RELATION = "depicts"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_claims(path: Path, article_relation: str) -> list[NamedTriple]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        (row["head"], row["relation"], row["tail"])
        for row in rows
        if row["relation"] not in {article_relation, _IMAGE_RELATION}
    ]


def build(cfg: QueryConfig) -> dict[str, Any]:
    """Write the retrieval task files and return the card."""
    places = sorted(_read_jsonl(cfg.core_dir / "places.jsonl"), key=lambda place: place["qid"])
    nodes: dict[str, dict[str, str]] = json.loads((cfg.core_dir / "kg_nodes.json").read_text(encoding="utf-8"))
    files_of: dict[str, list[str]] = {place["qid"]: [] for place in places}
    for image in _read_jsonl(cfg.core_dir / "images.jsonl"):
        files_of[image["place_qid"]].append(Path(image["path"]).name)

    labels = {place["qid"]: place["labels"] for place in places} | nodes
    names = {qid: first_label(node_labels, cfg.label_languages) for qid, node_labels in labels.items()}
    unnamed = sorted(qid for qid, name in names.items() if name is None)
    if unnamed:
        raise DataError(f"{len(unnamed)} nodes have no label in {cfg.label_languages}, e.g. {unnamed[:5]}")
    order = [place["qid"] for place in places] + sorted(nodes)
    entity_id = {qid: position for position, qid in enumerate(order)}
    entities = [
        {"entity_id": entity_id[qid], "qid": qid, "name": names[qid], "image_files": sorted(files_of.get(qid, []))}
        for qid in order
    ]

    place_ids = frozenset(files_of)
    claims = _read_claims(cfg.core_dir / "kg_triples_generic.tsv", cfg.article_relation)
    claims_of = split_claims(claims, place_ids, cfg.seed, cfg.val_fraction, cfg.test_fraction)
    mentions = [Mention(**row) for row in _read_jsonl(cfg.core_dir / "mention_evidence.jsonl")]
    articles_of = {
        scheme: article_triples(mention_triples(mentions, scheme, cfg.article_relation), cfg.article_prefix)
        for scheme in cfg.article_schemes
    }

    claim_relations = sorted({relation for _, relation, _ in claims})
    article_relations = sorted({relation for triples in articles_of.values() for _, relation, _ in triples})
    relation_id = {name: position for position, name in enumerate(claim_relations + article_relations)}
    relations = [
        {"relation_id": position, "pid": name, "label": name,
         "source": "article" if name in article_relations else "claim"}
        for name, position in relation_id.items()
    ]

    claim_ids = {split: to_ids(triples, entity_id, relation_id) for split, triples in claims_of.items()}
    article_ids = {scheme: to_ids(triples, entity_id, relation_id) for scheme, triples in articles_of.items()}
    templates = {relation_id[name]: template for name, template in cfg.templates.items() if name in relation_id}
    all_stats = direction_stats(claim_ids[Split.TRAIN], cfg.rules)
    stats = worded_directions(all_stats, templates)
    entity_names = [entity["name"] for entity in entities]
    indirect = build_indirect_queries(
        claim_ids, stats, templates, entity_names, [bool(e["image_files"]) for e in entities], cfg.rules.max_answers
    )
    # Hops are the same under every article scheme: the schemes rename edges, they do not move them.
    graph = claim_ids[Split.TRAIN] + article_ids[cfg.article_schemes[0]]
    hops = anchor_answer_hops(indirect, graph, cfg.rules.max_hops)
    indirect = [replace(query, hops=found) for query, found in zip(indirect, hops, strict=True)]
    nearby = triples_within_two_hops([q.anchor_id for q in indirect if q.anchor_id is not None], graph)
    strata = {
        q.query_id: {"triples_within_two_hops": nearby[q.anchor_id],
                     "tier": density_tier(nearby[q.anchor_id], cfg.sparse_max, cfg.medium_max).value}
        for q in indirect if q.anchor_id is not None
    }
    split_of_query = {q.query_id: q.split.value for q in indirect}
    place_split = {split: [entity_id[p["qid"]] for p in places if p["split"] == split.value] for split in Split}
    direct = build_direct_queries(place_split, entity_names, {})
    queries: list[EntityQuery] = direct + indirect

    card = {
        "entities": len(entities),
        "gallery_places": len(places),
        "images": sum(len(files) for files in files_of.values()),
        "claims": {split.value: len(triples) for split, triples in claim_ids.items()},
        "article_edges": {scheme.value: len(triples) for scheme, triples in article_ids.items()},
        "relations": {"claim": len(claim_relations),
                      **{scheme.value: len({r for _, r, _ in triples}) for scheme, triples in articles_of.items()}},
        "asked_directions": sorted(f"{relations[r]['pid']}/{d.value}" for (r, d), s in stats.items()
                                   if s.status.value == "kept"),
        "directions_too_small_or_unworded": sorted(
            f"{relations[r]['pid']}/{d.value}" for (r, d), s in all_stats.items()
            if (r, d) not in stats or s.status.value != "kept"),
        "queries": {f"{group}/{split}": count for (group, split), count in sorted(
            Counter((q.group.value, q.split.value) for q in queries).items())},
        "answers": {f"{group}/{split}": count for (group, split), count in sorted(
            Counter({(g, s): sum(len(q.relevant_ids) for q in queries if (q.group.value, q.split.value) == (g, s))
                     for g, s in {(q.group.value, q.split.value) for q in queries}}).items())},
        "indirect_density": {f"{split}/{tier}": count for (split, tier), count in sorted(
            Counter((split_of_query[query_id], row["tier"]) for query_id, row in strata.items()).items())},
        "indirect_hops": {f"{split.value}/{hops_ or 'none'}": count for (split, hops_), count in sorted(
            Counter((q.split, q.hops) for q in indirect).items(), key=str)},
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.out_dir / "entities.json", entities)
    write_json(cfg.out_dir / "relations.json", relations)
    write_json(cfg.out_dir / "triples.json", {
        **{split.value: [[t.head, t.relation, t.tail] for t in triples] for split, triples in claim_ids.items()},
        "articles": {scheme.value: [[t.head, t.relation, t.tail] for t in triples]
                     for scheme, triples in article_ids.items()},
    })
    write_json(cfg.out_dir / "queries.json", [query.to_dict() for query in queries])
    write_json(cfg.out_dir / "strata.json", strata)
    write_json(cfg.out_dir / "card.json", card)
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split-seed", type=int, help="other claim split; output goes to <out_dir>/split-<seed>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = QueryConfig.from_yaml(args.config)
        card = build(cfg if args.split_seed is None else cfg.with_split_seed(args.split_seed))
    except AppError:
        log.exception("build failed")
        return 1
    log.info("card: %s", json.dumps(card, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
