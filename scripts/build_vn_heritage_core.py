#!/usr/bin/env python3
"""Cut the dense core of the VN heritage graph and add administrative hierarchy edges.

Reads the full build (`data/processed/vn_heritage`) and the cached admin ancestry; no network.
Run from the repository root:
    python -m scripts.build_vn_heritage_core
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.connectors.wikidata import WikidataClient, WikipediaClient
from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import AppError, ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.vn_heritage_articles import (
    Cue,
    Mention,
    RelationScheme,
    body_link_records,
    collect_mentions,
    mention_triples,
    unresolved_shared_titles,
)
from src.data.vn_heritage_core import (
    CoreRules,
    Triple,
    admin_hierarchy,
    assign_split,
    century_node,
    claim_item_ids,
    drop_shadowed,
    inception_century,
    item_triples,
    select_core,
    split_groups,
)

log = logging.getLogger("build_vn_heritage_core")

DEFAULT_CONFIG = "configs/data/vn_heritage_core.yaml"
IMAGE_RELATION = "depicts"


@dataclass(frozen=True)
class CoreConfig:
    """Typed view of `configs/data/vn_heritage_core.yaml`."""

    source_dir: Path
    raw_dir: Path
    out_dir: Path
    rules: CoreRules
    seed: int
    val_fraction: float
    test_fraction: float
    extra_relations: dict[str, str]
    inception_property: str
    inception_relation: str
    wikidata_endpoint: str
    user_agent: str
    request_interval_s: float
    request_timeout_s: float
    article_language: str
    wikipedia_endpoint: str
    mention_relation: str
    stop_sections: frozenset[str]
    min_articles: int
    max_share: float
    class_property: str
    kind_of_class: dict[str, str]
    kind_of_role: dict[str, str]
    place_kind: str
    section_buckets: dict[str, str]
    skip_after: tuple[str, ...]
    cues: tuple[Cue, ...]
    relation_scheme: RelationScheme
    ablation_schemes: tuple[RelationScheme, ...]
    count_mentions_toward_selection: bool

    @classmethod
    def from_yaml(cls, path: str) -> CoreConfig:
        """Load and validate the config; a missing or mistyped field raises ConfigError."""
        raw = read_mapping(path)
        paths, selection = require_section(raw, "paths"), require_section(raw, "selection")
        hierarchy, split = require_section(raw, "hierarchy"), require_section(raw, "split")
        extra = require_section(raw, "extra")
        articles = require_section(raw, "articles")
        rules = CoreRules(
            min_triples=require_field(selection, "min_triples", int),
            min_images=require_field(selection, "min_images", int),
            label_language=require_field(selection, "label_language", str),
            class_relations=frozenset(require_field(selection, "class_relations", list)),
            shadowed_relations=dict(require_section(selection, "shadowed_relations")),
            admin_relations=frozenset(require_field(hierarchy, "admin_relations", list)),
            hierarchy_relation=require_field(hierarchy, "relation", str),
            group_relations=frozenset(require_field(split, "group_relations", list)),
            province_classes=frozenset(require_field(hierarchy, "province_classes", list)),
        )
        return cls(
            source_dir=Path(require_field(paths, "source_dir", str)),
            raw_dir=Path(require_field(paths, "raw_dir", str)),
            out_dir=Path(require_field(paths, "out_dir", str)),
            rules=rules,
            seed=require_field(split, "seed", int),
            val_fraction=require_field(split, "val_fraction", float),
            test_fraction=require_field(split, "test_fraction", float),
            extra_relations=dict(require_section(extra, "relations")),
            inception_property=require_field(extra, "inception_property", str),
            inception_relation=require_field(extra, "inception_relation", str),
            wikidata_endpoint=require_field(extra, "wikidata_endpoint", str),
            user_agent=require_field(extra, "user_agent", str),
            request_interval_s=require_field(extra, "request_interval_s", float),
            request_timeout_s=require_field(extra, "request_timeout_s", float),
            article_language=require_field(articles, "language", str),
            wikipedia_endpoint=require_field(articles, "endpoint", str),
            mention_relation=require_field(articles, "relation", str),
            stop_sections=frozenset(require_field(articles, "stop_sections", list)),
            min_articles=require_field(articles, "min_articles", int),
            max_share=require_field(articles, "max_share", float),
            class_property=require_field(articles, "class_property", str),
            kind_of_class=dict(require_section(articles, "kind_of_class")),
            kind_of_role=dict(require_section(articles, "kind_of_role")),
            place_kind=require_field(articles, "place_kind", str),
            section_buckets=dict(require_section(articles, "section_buckets")),
            skip_after=tuple(require_field(articles, "skip_after", list)),
            cues=tuple(_parse_cue(raw_cue) for raw_cue in require_field(articles, "cues", list)),
            relation_scheme=_parse_scheme(require_field(articles, "relation_scheme", str)),
            ablation_schemes=tuple(_parse_scheme(v) for v in require_field(articles, "ablation_schemes", list)),
            count_mentions_toward_selection=_require_bool(articles, "count_toward_selection"),
        )


def _parse_cue(raw: Any) -> Cue:
    if not isinstance(raw, dict):
        raise ConfigError(f"each cue must be a mapping, got {raw!r}")
    return Cue(
        relation=require_field(raw, "relation", str),
        keywords=tuple(require_field(raw, "keywords", list)),
        kinds=frozenset(require_field(raw, "kinds", list)),
    )


def _parse_scheme(value: str) -> RelationScheme:
    try:
        return RelationScheme(value)
    except ValueError as error:
        known = [scheme.value for scheme in RelationScheme]
        raise ConfigError(f"unknown relation scheme {value!r}; use one of {known}") from error


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"field '{key}' must be true or false, got {value!r}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_triples(path: Path) -> list[Triple]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return [(head, relation, tail) for head, relation, tail in (line.split("\t") for line in lines)]


def _labels_of(entities: dict[str, dict[str, Any]], languages: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Labels of cached Wikidata entities in the wanted languages; entities without any are left out."""
    found = {
        qid: {lang: entity["labels"][lang]["value"] for lang in languages if lang in entity.get("labels", {})}
        for qid, entity in entities.items()
    }
    return {qid: names for qid, names in found.items() if names}


def _extra_labels(cfg: CoreConfig, wanted: set[str], languages: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Labels of extra target items; anything not yet in the cache file is fetched and stored."""
    cache_file = cfg.raw_dir / "extra_labels.json"
    cached: dict[str, dict[str, str]] = read_json(cache_file) if cache_file.exists() else {}
    missing = wanted - cached.keys()
    if missing:
        log.info("fetching labels of %d items from %s", len(missing), cfg.wikidata_endpoint)
        client = WikidataClient(cfg.wikidata_endpoint, cfg.user_agent, cfg.request_interval_s, cfg.request_timeout_s)
        fetched = client.labels(missing, languages)
        # Items without a label are cached as empty so they are not requested on every build.
        cached.update({qid: fetched.get(qid, {}) for qid in missing})
        write_json(cache_file, cached)
    return {qid: names for qid, names in cached.items() if qid in wanted and names}


def _extra_knowledge(
    cfg: CoreConfig, entities: dict[str, dict[str, Any]], labels: dict[str, dict[str, str]],
    languages: tuple[str, ...],
) -> tuple[list[Triple], int]:
    """Extra triples of all places; adds the labels of their targets to `labels` in place.

    Returns the triples and how many were dropped because the target has no label.
    """
    triples = item_triples(entities, cfg.extra_relations)
    labels.update(_extra_labels(cfg, {tail for _, _, tail in triples} - labels.keys(), languages))
    labelled = [t for t in triples if t[2] in labels]
    dropped = len(triples) - len(labelled)
    for qid in sorted(entities):
        century = inception_century(entities[qid], cfg.inception_property)
        if century is not None:
            node, names = century_node(century)
            labels[node] = names
            labelled.append((qid, cfg.inception_relation, node))
    return labelled, dropped


def _fill_cache(path: Path, wanted: set[str], fetch: Callable[[set[str]], dict[str, str]]) -> dict[str, str]:
    """Read a {key: text} cache file, fetch the keys it lacks and store them ("" when absent upstream)."""
    cached: dict[str, str] = read_json(path) if path.exists() else {}
    missing = wanted - cached.keys()
    if missing:
        log.info("fetching %d entries for %s", len(missing), path.name)
        fetched = fetch(missing)
        cached.update({key: fetched.get(key, "") for key in missing})
        write_json(path, cached, indent=None)
    return cached


def _target_kinds(
    cfg: CoreConfig, wikidata: WikidataClient, places: dict[str, dict[str, Any]],
    entities: dict[str, dict[str, Any]], knowledge: list[Triple], candidates: set[str],
) -> dict[str, str]:
    """Kind of each mention target that has one: a place, a known Wikidata class, or its graph role."""
    outside = {qid for qid in candidates if qid.startswith("Q") and qid not in entities}
    fetched = _fill_cache(
        cfg.raw_dir / "link_item_classes.json", outside,
        lambda qids: {q: "|".join(c) for q, c in wikidata.claim_items(qids, cfg.class_property).items()},
    )
    pointing = Counter((tail, relation) for _, relation, tail in knowledge)
    role: dict[str, str] = {}
    for (tail, relation), _ in pointing.most_common():
        role.setdefault(tail, relation)
    kinds: dict[str, str] = {}
    for qid in candidates:
        if qid in places:
            kinds[qid] = cfg.place_kind
            continue
        classes = claim_item_ids(entities[qid], cfg.class_property) if qid in entities else fetched[qid].split("|")
        kind = next((cfg.kind_of_class[c] for c in classes if c in cfg.kind_of_class), None)
        kind = kind or cfg.kind_of_role.get(role.get(qid, ""))
        if kind:
            kinds[qid] = kind
    return kinds


def _article_mentions(
    cfg: CoreConfig, places: dict[str, dict[str, Any]], entities: dict[str, dict[str, Any]],
    knowledge: list[Triple], labels: dict[str, dict[str, str]],
) -> list[Mention]:
    """Mentions taken from the body links of each place's article; labels new targets in place."""
    site = f"{cfg.article_language}wiki"
    wikipedia = WikipediaClient(cfg.wikipedia_endpoint, cfg.user_agent, cfg.request_interval_s, cfg.request_timeout_s)
    wikidata = WikidataClient(cfg.wikidata_endpoint, cfg.user_agent, cfg.request_interval_s, cfg.request_timeout_s)
    article = {qid: place["sitelinks"][site] for qid, place in places.items() if site in place["sitelinks"]}
    wikitext = _fill_cache(cfg.raw_dir / "article_wikitext.json", set(article.values()), wikipedia.wikitext)
    links = {
        qid: body_link_records(wikitext[title], cfg.stop_sections, cfg.section_buckets)
        for qid, title in article.items()
    }

    graph_nodes = set(places) | {tail for _, _, tail in knowledge}
    title_to_qid = {
        entity["sitelinks"][site]["title"]: qid
        for qid, entity in entities.items()
        if site in entity.get("sitelinks", {})
    }
    outside = {node for node in graph_nodes if node.startswith("Q") and node not in entities}
    node_titles = _fill_cache(
        cfg.raw_dir / "node_sitelinks.json", outside, lambda qids: wikidata.sitelink_titles(qids, site)
    )
    title_to_qid.update({title: qid for qid, title in node_titles.items() if title and qid in outside})
    shared = unresolved_shared_titles(links, title_to_qid, cfg.min_articles)
    link_items = _fill_cache(cfg.raw_dir / "article_link_items.json", shared, wikipedia.page_items)
    title_to_qid.update({title: qid for title, qid in link_items.items() if qid and title in shared})

    candidates = {title_to_qid[r.title] for records in links.values() for r in records if r.title in title_to_qid}
    kinds = _target_kinds(cfg, wikidata, places, entities, knowledge, candidates)
    mentions = collect_mentions(
        links, title_to_qid, kinds, graph_nodes, {(head, tail) for head, _, tail in knowledge},
        cfg.cues, cfg.skip_after, cfg.min_articles, cfg.max_share,
    )
    # A target that only Wikipedia introduced is labelled with its article title.
    for title, qid in sorted(title_to_qid.items()):
        labels.setdefault(qid, {cfg.article_language: title})
    return mentions


def build(cfg: CoreConfig) -> dict[str, Any]:
    """Write the core dataset and return its statistics."""
    rules = cfg.rules
    places = {row["qid"]: row for row in _read_jsonl(cfg.source_dir / "places.jsonl")}
    images = _read_jsonl(cfg.source_dir / "images.jsonl")
    node_labels: dict[str, dict[str, str]] = read_json(cfg.source_dir / "kg_nodes.json")
    knowledge = [t for t in _read_triples(cfg.source_dir / "kg_triples.tsv") if t[1] != IMAGE_RELATION]
    knowledge = drop_shadowed(knowledge, rules.shadowed_relations)
    languages = tuple(next(iter(node_labels.values())))
    entities: dict[str, dict[str, Any]] = read_json(cfg.raw_dir / "entities.json")
    labels = {**_labels_of(entities, languages), **node_labels}
    extra, extra_unlabelled = _extra_knowledge(cfg, {q: entities[q] for q in places}, labels, languages)
    known = set(knowledge)
    knowledge += [t for t in extra if t not in known]
    mentions = _article_mentions(cfg, places, entities, knowledge, labels)
    base = cfg.mention_relation

    image_counts = Counter(image["place_qid"] for image in images)
    counted = knowledge + (mention_triples(mentions, RelationScheme.GENERIC, base)
                           if cfg.count_mentions_toward_selection else [])
    core = set(select_core({q: p["labels"] for q, p in places.items()}, counted, image_counts, rules))
    if not core:
        raise DataError("no place passes the core selection; lower min_triples or min_images")
    claims = [t for t in knowledge if t[0] in core]
    core_mentions = [m for m in mentions if m.head in core]

    admin_graph = read_json(cfg.raw_dir / "admin_graph.json")
    parents: dict[str, list[str]] = admin_graph["parents"]
    admin_units = {tail for _, relation, tail in claims if relation in rules.admin_relations}
    hierarchy, skipped = admin_hierarchy(
        admin_units, parents, admin_graph["classes"], rules.province_classes,
        {qid for qid, names in labels.items() if any(names.values())}, rules.hierarchy_relation,
    )
    existing = set(claims)
    hierarchy = [edge for edge in hierarchy if edge not in existing]

    groups = split_groups(core, claims, rules.group_relations)
    core_places = []
    for qid in sorted(core):
        split = assign_split(groups[qid], cfg.seed, cfg.val_fraction, cfg.test_fraction)
        core_places.append({**places[qid], "split": split.value})
    core_images = [image for image in images if image["place_qid"] in core]
    image_edges = [(image["image_id"], IMAGE_RELATION, image["place_qid"]) for image in core_images]
    node_ids = {tail for _, _, tail in claims + hierarchy} | {head for head, _, _ in hierarchy}
    node_ids |= {mention.tail for mention in core_mentions}

    missing = node_ids - labels.keys() - core
    if missing:
        raise DataError(f"knowledge nodes without a label: {sorted(missing)[:5]}")

    _write_jsonl(cfg.out_dir / "places.jsonl", core_places)
    _write_jsonl(cfg.out_dir / "images.jsonl", core_images)
    _write_jsonl(cfg.out_dir / "mention_evidence.jsonl", [mention.to_dict() for mention in core_mentions])
    write_json(cfg.out_dir / "kg_nodes.json", {qid: labels[qid] for qid in sorted(node_ids - core)})
    # The same graph under each relation scheme, so an ablation only swaps the triple file.
    for scheme in dict.fromkeys((cfg.relation_scheme, *cfg.ablation_schemes)):
        triples = claims + hierarchy + mention_triples(core_mentions, scheme, base) + image_edges
        text = "head\trelation\ttail\n" + "".join(f"{h}\t{r}\t{t}\n" for h, r, t in triples)
        (cfg.out_dir / f"kg_triples_{scheme.value}.tsv").write_text(text, encoding="utf-8")
        if scheme is cfg.relation_scheme:
            (cfg.out_dir / "kg_triples.tsv").write_text(text, encoding="utf-8")
    core_triples = claims + mention_triples(core_mentions, cfg.relation_scheme, base)
    stats = _stats(
        core_places, core_images, core_triples, hierarchy, node_ids, groups, skipped, extra_unlabelled, rules
    )
    stats["claim_triples"] = len(claims)
    stats["mentions"] = {
        "total": len(core_mentions),
        "scheme": cfg.relation_scheme.value,
        "by_kind": dict(Counter(m.kind for m in core_mentions)),
        "by_section": dict(Counter(m.section for m in core_mentions)),
        "by_cue": dict(Counter(m.cue or "none" for m in core_mentions)),
    }
    write_json(cfg.out_dir / "stats.json", stats)
    return stats


def _stats(
    places: list[dict[str, Any]], images: list[dict[str, Any]], core_triples: list[Triple],
    hierarchy: list[Triple], node_ids: set[str], groups: dict[str, str], skipped: int,
    extra_unlabelled: int, rules: CoreRules,
) -> dict[str, Any]:
    split_of = {place["qid"]: place["split"] for place in places}
    degree = Counter(head for head, _, _ in core_triples)
    return {
        "places": len(places),
        "knowledge_nodes": len(node_ids - split_of.keys()),
        "entities": len(node_ids | split_of.keys()),
        "images": len(images),
        "knowledge_triples": len(core_triples) + len(hierarchy),
        "place_triples": len(core_triples),
        "hierarchy_triples": len(hierarchy),
        "hierarchy_parents_without_label": skipped,
        "extra_triples_without_label": extra_unlabelled,
        "relations": dict(Counter(relation for _, relation, _ in core_triples + hierarchy)),
        "places_by_split": dict(Counter(split_of.values())),
        "images_by_split": dict(Counter(split_of[image["place_qid"]] for image in images)),
        "places_by_group": dict(Counter(place["group"] for place in places)),
        "triples_per_place_min": min(degree[qid] for qid in split_of),
        "triples_per_place_mean": round(len(core_triples) / len(places), 2),
        "split_groups_with_several_places": sum(1 for n in Counter(groups.values()).values() if n > 1),
        "selection": {"min_triples": rules.min_triples, "min_images": rules.min_images,
                      "label_language": rules.label_language},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; the single place where errors are logged."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", help="override paths.raw_dir (e.g. the raw cache of another checkout)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = CoreConfig.from_yaml(args.config)
        if args.raw_dir:
            cfg = CoreConfig(**{**vars(cfg), "raw_dir": Path(args.raw_dir)})
        stats = build(cfg)
    except AppError:
        log.exception("core build failed")
        return 1
    log.info("core: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
