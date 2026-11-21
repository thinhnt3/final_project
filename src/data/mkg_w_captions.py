"""MKG-W reshaped like VG ∩ COCO: one item per entity with an image, captions and its own graph.

An item is an entity that has images, a description and enough triples. Its captions are the
sentences of its description with the entity's name masked; its graph is the set of triples
that touch it, with the entity itself replaced by one shared "this entity" node. The files
written by `build_mkg_w_captions` have the layout the VG ∩ COCO retrieval scripts read.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.data.mkg_w_raw import Triple, display_name, split_sentences
from src.data.schemas import Split

# Concept id of the item's own entity in every item graph. Its text never names the entity.
CENTER_CONCEPT = 0
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)$")


@dataclass(frozen=True)
class CaptionSubsetConfig:
    """Rules of the subset; see the comments in `configs/data/mkg_w_captions.yaml`."""

    source_config: Path
    min_triples: int
    captions_per_item: int
    min_caption_chars: int
    mask_text: str
    fractions: dict[Split, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> CaptionSubsetConfig:
        """Read the `subset` section; raise ConfigError on any problem."""
        raw = require_section(read_mapping(path), "subset")
        fractions_raw = require_section(raw, "fractions")
        fractions = {s: float(require_field(fractions_raw, s.value, (int, float))) for s in Split}
        config = cls(
            source_config=Path(require_field(raw, "source_config", str)),
            min_triples=require_field(raw, "min_triples", int),
            captions_per_item=require_field(raw, "captions_per_item", int),
            min_caption_chars=require_field(raw, "min_caption_chars", int),
            mask_text=require_field(raw, "mask_text", str),
            fractions=fractions,
        )
        if config.min_triples < 1 or config.captions_per_item < 1 or config.min_caption_chars < 1:
            raise ConfigError("subset.min_triples, captions_per_item and min_caption_chars must be >= 1")
        if not config.mask_text.strip() or abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ConfigError("subset.mask_text must not be empty and subset.fractions must sum to 1")
        return config


def masked_captions(description: str, name: str, config: CaptionSubsetConfig) -> list[str]:
    """The first `captions_per_item` usable sentences of the description with the entity's name replaced by the mask.

    Both the full name and the name without its trailing parenthesis ("Signs (film)" → "Signs")
    are masked, so a caption cannot be matched to its item by the title alone. Sentences
    shorter than `min_caption_chars` after masking are dropped.
    """
    shown = display_name(name)
    names = sorted({shown, _PARENTHETICAL.sub("", shown)}, key=len, reverse=True)
    captions: list[str] = []
    for sentence in split_sentences(description):
        for variant in names:
            sentence = sentence.replace(variant, config.mask_text)
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        if len(sentence) >= config.min_caption_chars and sentence not in captions:
            captions.append(sentence)
        if len(captions) == config.captions_per_item:
            break
    return captions


def select_items(entities: Sequence[Mapping[str, Any]], triples: Sequence[Triple],
                 config: CaptionSubsetConfig) -> dict[int, list[str]]:
    """Entity id → captions, for entities with an image, `captions_per_item` captions and `min_triples` triples.

    Every item has the same number of captions, as in COCO; the training code relies on it.
    """
    degree: Counter[int] = Counter()
    for triple in triples:
        degree[triple.head] += 1
        degree[triple.tail] += 1
    items: dict[int, list[str]] = {}
    for entity in entities:
        entity_id = entity["entity_id"]
        if entity["image_files"] and entity["description"] and degree[entity_id] >= config.min_triples:
            captions = masked_captions(entity["description"], entity["name"], config)
            if len(captions) == config.captions_per_item:
                items[entity_id] = captions
    return items


def item_triples(triples: Sequence[Triple]) -> dict[int, list[Triple]]:
    """Entity id → the triples that touch it, in input order."""
    touching: dict[int, list[Triple]] = {}
    for triple in triples:
        for entity_id in {triple.head, triple.tail}:
            touching.setdefault(entity_id, []).append(triple)
    return touching


def build_vocabulary(
    item_split: Mapping[int, Split],
    touching: Mapping[int, Sequence[Triple]],
    names: Sequence[str],
    relation_labels: Sequence[str],
    mask_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int], dict[int, int]]:
    """Concept and predicate tables, plus entity id → concept id and relation id → predicate id.

    A concept is a neighbouring entity, named in full; concept 0 is the masked item itself.
    Both tables are ordered by the number of train items they occur in. Every neighbour of
    every item is kept: node features are frozen CLIP text vectors, so a neighbour seen only
    in validation or test needs no trained parameter. `idf` uses train counts only.
    """
    train_items = [i for i, split in item_split.items() if split is Split.TRAIN]
    concept_train: Counter[int] = Counter()
    predicate_train: Counter[int] = Counter()
    for item in train_items:
        concept_train.update({e for t in touching[item] for e in (t.head, t.tail) if e != item})
        predicate_train.update({t.relation for t in touching[item]})
    neighbours = {e for item in item_split for t in touching[item] for e in (t.head, t.tail) if e != item}
    relations = {t.relation for item in item_split for t in touching[item]}

    def idf(count: int) -> float:
        return round(math.log((len(train_items) + 1) / (count + 1)) + 1, 6)

    concepts = [{"concept_id": CENTER_CONCEPT, "key": "center", "text": mask_text,
                 "train_images": len(train_items), "idf": idf(len(train_items))}]
    concept_of: dict[int, int] = {}
    for entity_id in sorted(neighbours, key=lambda e: (-concept_train[e], e)):
        concept_of[entity_id] = len(concepts)
        concepts.append({"concept_id": len(concepts), "key": f"entity-{entity_id}",
                         "text": display_name(names[entity_id]), "train_images": concept_train[entity_id],
                         "idf": idf(concept_train[entity_id])})
    predicate_of = {r: i for i, r in enumerate(sorted(relations, key=lambda r: (-predicate_train[r], r)))}
    predicates = [{"predicate_id": i, "text": relation_labels[r], "train_images": predicate_train[r]}
                  for r, i in predicate_of.items()]
    return concepts, predicates, concept_of, predicate_of


def item_graph(item: int, split: Split, touching: Sequence[Triple], concept_of: Mapping[int, int],
               predicate_of: Mapping[int, int]) -> dict[str, Any]:
    """The item's graph in the VG ∩ COCO `image_graphs.json` shape; the item itself is the center concept."""
    def concept(entity_id: int) -> int:
        return CENTER_CONCEPT if entity_id == item else concept_of[entity_id]

    triplets = sorted({(concept(t.head), predicate_of[t.relation], concept(t.tail)) for t in touching})
    concept_ids = sorted({CENTER_CONCEPT, *(c for s, _, o in triplets for c in (s, o))})
    # Each entity is one node, unlike a scene graph where a concept can label several objects.
    return {"vg_image_id": item, "split": split.value, "concept_ids": concept_ids,
            "concept_counts": [1] * len(concept_ids), "triplets": [list(t) for t in triplets]}
