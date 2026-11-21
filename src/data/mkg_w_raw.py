"""Readers for the raw MKG-W files: id maps, triples, entity names and descriptions."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from src.core.exceptions import DataError

_DBPEDIA_PREFIX = "http://dbpedia.org/resource/"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Shorter cuts are usually abbreviations ("Dr. ", "St. "), not the end of the first sentence.
_MIN_SENTENCE_CHARS = 40
_INITIAL_END = re.compile(r"(?:^|\s)[A-Z]\.$")


@dataclass(frozen=True)
class Triple:
    """One fact: head entity id, relation id, tail entity id."""

    head: int
    relation: int
    tail: int


def read_id_file(path: Path) -> list[str]:
    """Read `entity2id.txt` or `relation2id.txt`; return the last URI segment of each id, in id order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in lines[1:] if line.strip()]
    if not lines or int(lines[0]) != len(rows):
        raise DataError(f"{path}: header count does not match {len(rows)} rows")
    keys = [""] * len(rows)
    for row in rows:
        if len(row) != 2 or not 0 <= int(row[1]) < len(rows) or keys[int(row[1])]:
            raise DataError(f"{path}: bad or duplicate id row {row}")
        keys[int(row[1])] = row[0].rsplit("/", 1)[-1]
    return keys


def read_triples(path: Path, n_entities: int, n_relations: int) -> list[Triple]:
    """Read a `*2id.txt` triple file. Each row is `head tail relation` (OpenKE column order)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    triples = [Triple(int(h), int(r), int(t)) for h, t, r in (line.split() for line in lines[1:] if line.strip())]
    if not lines or int(lines[0]) != len(triples):
        raise DataError(f"{path}: header count does not match {len(triples)} rows")
    for triple in triples:
        if not (0 <= triple.head < n_entities and 0 <= triple.tail < n_entities and 0 <= triple.relation < n_relations):
            raise DataError(f"{path}: id out of range in {triple}")
    return triples


def normalize_name(name: str) -> str:
    """Canonical form of a DBpedia resource name: percent-decoded, NFC."""
    return unicodedata.normalize("NFC", unquote(name))


def loose_key(name: str) -> str:
    """Letters and digits only, lowercased; equal for names that differ in punctuation."""
    return "".join(ch for ch in normalize_name(name).lower() if ch.isalnum())


def display_name(name: str) -> str:
    """Readable entity name for query text: underscores become spaces."""
    return normalize_name(name).replace("_", " ").strip()


def read_entity_names(path: Path, qids: Iterable[str]) -> list[str]:
    """Read `ent_links` (DBpedia URI, Wikidata URI per row); return the DBpedia name of each qid, in order."""
    by_qid: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        dbpedia_uri, wikidata_uri = line.split("\t")
        by_qid[wikidata_uri.rsplit("/", 1)[-1]] = normalize_name(dbpedia_uri.removeprefix(_DBPEDIA_PREFIX))
    qid_list = list(qids)
    missing = [qid for qid in qid_list if qid not in by_qid]
    if missing:
        raise DataError(f"{path}: {len(missing)} entities have no name, e.g. {missing[:5]}")
    return [by_qid[qid] for qid in qid_list]


def read_descriptions(path: Path) -> dict[str, str]:
    """Read the description file (`<DBpedia URI>\\t"text"@en .` per row), keyed by normalized name."""
    descriptions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        uri, _, literal = line.partition("\t")
        start, end = literal.find('"'), literal.rfind('"')
        if not uri.startswith(_DBPEDIA_PREFIX) or end <= start:
            raise DataError(f"{path}: unexpected row {line[:80]!r}")
        text = literal[start + 1:end].replace('\\"', '"').strip()
        if text:
            descriptions[normalize_name(uri.removeprefix(_DBPEDIA_PREFIX))] = text
    return descriptions


def first_sentence(text: str, max_chars: int = 300) -> str:
    """First sentence of a description, cut at `max_chars` on a word boundary when longer."""
    sentence = text
    for match in _SENTENCE_END.finditer(text):
        if match.start() >= _MIN_SENTENCE_CHARS:
            sentence = text[:match.start()]
            break
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0]
    return sentence.strip()


def split_sentences(text: str, min_chars: int = _MIN_SENTENCE_CHARS) -> list[str]:
    """Sentences of a description; a piece that is not a whole sentence is joined to the next one.

    A piece shorter than `min_chars` is usually cut at an abbreviation ("Dr. ", "St. "), and
    one ending in a single capital letter at a name initial ("Edward G. Robinson").
    """
    sentences: list[str] = []
    pending = ""
    for piece in _SENTENCE_END.split(text.strip()):
        pending = f"{pending} {piece}".strip()
        if len(pending) >= min_chars and not _INITIAL_END.search(pending):
            sentences.append(pending)
            pending = ""
    if pending:
        sentences.append(pending)
    return sentences


def match_folders(names: Mapping[int, str], folders: Iterable[str]) -> dict[int, str]:
    """Map entity id → image folder name.

    Exact match on the normalized name first. The archive rewrites some punctuation in
    folder names (`:` became `__`), so a second pass matches on `loose_key` and accepts
    only pairs where one remaining entity and one remaining folder share the key.
    """
    by_name: dict[str, str] = {normalize_name(folder): folder for folder in folders}
    matched = {entity_id: by_name[name] for entity_id, name in names.items() if name in by_name}
    used = set(matched.values())
    loose_folders: dict[str, list[str]] = {}
    for folder in by_name.values():
        if folder not in used:
            loose_folders.setdefault(loose_key(folder), []).append(folder)
    loose_entities: dict[str, list[int]] = {}
    for entity_id, name in names.items():
        if entity_id not in matched:
            loose_entities.setdefault(loose_key(name), []).append(entity_id)
    for key, entity_ids in loose_entities.items():
        candidates = loose_folders.get(key, [])
        if key and len(entity_ids) == 1 and len(candidates) == 1:
            matched[entity_ids[0]] = candidates[0]
    return matched
