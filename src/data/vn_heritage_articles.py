"""Pure transforms turning Wikipedia article source into typed mention edges between entities.

Only links written in the article body count. Navigation boxes are templates, so dropping
templates removes the links every article of a series shares with every other one.

A mention is described three ways (kind of the target, section of the article, cue word in
the sentence); a scheme picks which description names the relation, so the same mentions can
be exported under several schemes and compared in an ablation.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

Triple = tuple[str, str, str]

_LINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]*))?\]\]")
_REF_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^(==+)\s*(.*?)\s*==+\s*$", re.MULTILINE)
_MARKUP_RE = re.compile(r"'{2,}|\[\[|\]\]")
_SENTENCE_END = ".!?\n"
_WINDOW = 90  # characters of the sentence kept on each side of a link

LEAD_SECTION = "lead"
OTHER_SECTION = "other"


class RelationScheme(StrEnum):
    """How a mention is turned into a relation name."""

    GENERIC = "generic"   # one relation for every mention
    KIND = "kind"         # by the kind of the target (person, era, ...)
    SECTION = "section"   # by the article section holding the link
    CUE = "cue"           # by the cue word in the sentence, else by kind


@dataclass(frozen=True)
class LinkRecord:
    """One body link: target title, section bucket and the sentence text around it."""

    title: str
    section: str
    before: str
    after: str


@dataclass(frozen=True)
class Cue:
    """Words that, next to a link to a target of one of `kinds`, state a relation."""

    relation: str
    keywords: tuple[str, ...]
    kinds: frozenset[str]


@dataclass(frozen=True)
class Mention:
    """An accepted (article entity -> linked entity) pair with everything known about it."""

    head: str
    tail: str
    kind: str
    section: str
    cue: str | None
    context: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the evidence file."""
        return asdict(self)


def strip_templates(text: str) -> str:
    """Remove `{{...}}` blocks, including nested ones."""
    kept: list[str] = []
    depth = index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair == "{{":
            depth += 1
            index += 2
        elif pair == "}}" and depth:
            depth -= 1
            index += 2
        else:
            if not depth:
                kept.append(text[index])
            index += 1
    return "".join(kept)


def body_text(wikitext: str, stop_sections: frozenset[str]) -> str:
    """Article prose: no comments, references or templates, cut at the first reference-type section."""
    text = strip_templates(_REF_RE.sub(" ", _COMMENT_RE.sub(" ", wikitext)))
    for heading in _HEADING_RE.finditer(text):
        if heading.group(2).casefold() in stop_sections:
            return text[: heading.start()]
    return text


def normalise_title(raw: str) -> str:
    """MediaWiki title form: underscores are spaces and the first letter is upper case."""
    title = " ".join(raw.replace("_", " ").split())
    return title[:1].upper() + title[1:]


def _plain(text: str) -> str:
    """Display text of a wikitext fragment: links replaced by what the reader sees."""
    shown = _LINK_RE.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(1), text)
    return " ".join(_MARKUP_RE.sub("", shown).split())


def _sentence_before(text: str, end: int) -> str:
    start = max(max(text.rfind(mark, 0, end) for mark in _SENTENCE_END) + 1, end - _WINDOW, 0)
    return _plain(text[start:end])


def _sentence_after(text: str, start: int) -> str:
    stops = [pos for pos in (text.find(mark, start) for mark in _SENTENCE_END) if pos != -1]
    return _plain(text[start : min([*stops, start + _WINDOW])])


def body_link_records(
    wikitext: str, stop_sections: frozenset[str], section_buckets: dict[str, str]
) -> list[LinkRecord]:
    """Main-namespace body links in reading order, each with its section bucket and sentence.

    `section_buckets` maps a word found in a heading to a bucket name. Text before the first
    heading is the lead; a top-level heading with no known word starts the "other" bucket and
    a deeper one keeps the bucket of its parent.
    """
    text = body_text(wikitext, stop_sections)
    headings = []
    bucket = LEAD_SECTION
    for heading in _HEADING_RE.finditer(text):
        name = heading.group(2).casefold()
        matched = next((value for word, value in section_buckets.items() if word in name), None)
        if matched is not None:
            bucket = matched
        elif len(heading.group(1)) == 2:
            bucket = OTHER_SECTION
        headings.append((heading.start(), bucket))
    records = []
    for link in _LINK_RE.finditer(text):
        title = normalise_title(link.group(1))
        # A colon marks a namespace or interwiki prefix (file, category, another language).
        if not title or ":" in title:
            continue
        section = next((name for start, name in reversed(headings) if start < link.start()), LEAD_SECTION)
        records.append(
            LinkRecord(title, section, _sentence_before(text, link.start()), _sentence_after(text, link.end()))
        )
    return records


def unresolved_shared_titles(
    links: dict[str, list[LinkRecord]], title_to_qid: dict[str, str], min_articles: int
) -> set[str]:
    """Link titles with no known item that at least `min_articles` articles use."""
    usage = Counter(
        title for records in links.values() for title in {r.title for r in records} if title not in title_to_qid
    )
    return {title for title, count in usage.items() if count >= min_articles}


def _keyword_spans(text: str, keyword: str) -> list[tuple[int, int]]:
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return [match.span() for match in re.finditer(pattern, text.casefold())]


def follows_any(before: str, words: tuple[str, ...]) -> bool:
    """True when the text right before a link ends with one of `words` (e.g. "đường")."""
    tail = before.casefold().rstrip(" \"'“(")
    return any(re.search(rf"(?<!\w){re.escape(word)}$", tail) for word in words)


def cue_relation(before: str, after: str, kind: str, cues: tuple[Cue, ...]) -> str | None:
    """Relation stated by the cue word closest to the link, or None.

    The words before the link are searched first (closest wins), then the words after it;
    a cue only applies to targets of its kinds.
    """
    usable = [cue for cue in cues if kind in cue.kinds]
    best: tuple[int, str] | None = None
    for cue in usable:
        for keyword in cue.keywords:
            for _, end in _keyword_spans(before, keyword):
                if best is None or end > best[0]:
                    best = (end, cue.relation)
    if best is not None:
        return best[1]
    for cue in usable:
        for keyword in cue.keywords:
            for start, _ in _keyword_spans(after, keyword):
                if best is None or start < best[0]:
                    best = (start, cue.relation)
    return best[1] if best is not None else None


def collect_mentions(
    links: dict[str, list[LinkRecord]],
    title_to_qid: dict[str, str],
    kinds: dict[str, str],
    graph_nodes: set[str],
    connected: set[tuple[str, str]],
    cues: tuple[Cue, ...],
    skip_after: tuple[str, ...],
    min_articles: int,
    max_share: float,
) -> list[Mention]:
    """One mention per (article entity, linked entity) pair that passes every filter.

    `links` maps an entity to the link records of its article. A link is dropped when it
    follows one of `skip_after` (a street named after a person is not that person), when its
    target has no kind (common nouns and classes), when the target is linked from more than
    `max_share` of the articles ("Việt Nam"), when a target outside the graph is linked from
    fewer than `min_articles` articles, or when the pair is already joined by another edge.
    The first link to a target in reading order supplies section, cue and context.
    """
    first: dict[tuple[str, str], LinkRecord] = {}
    for source, records in links.items():
        for record in records:
            target = title_to_qid.get(record.title)
            if target is None or target == source or target not in kinds:
                continue
            if follows_any(record.before, skip_after):
                continue
            first.setdefault((source, target), record)
    usage = Counter(target for _, target in first)
    ceiling = max_share * len(links)
    mentions = []
    for (source, target), record in sorted(first.items()):
        if usage[target] > ceiling or (target not in graph_nodes and usage[target] < min_articles):
            continue
        if (source, target) in connected or (target, source) in connected:
            continue
        kind = kinds[target]
        mentions.append(Mention(
            head=source, tail=target, kind=kind, section=record.section,
            cue=cue_relation(record.before, record.after, kind, cues),
            context=f"{record.before} 【{record.title}】 {record.after}".strip(),
        ))
    return mentions


def mention_relation(mention: Mention, scheme: RelationScheme, base: str) -> str:
    """Relation name of a mention under one scheme; `base` is the generic relation name."""
    if scheme is RelationScheme.GENERIC:
        return base
    if scheme is RelationScheme.SECTION:
        return f"{mention.section}_{base}"
    if scheme is RelationScheme.CUE and mention.cue is not None:
        return mention.cue
    return f"{base}_{mention.kind}"


def mention_triples(mentions: list[Mention], scheme: RelationScheme, base: str) -> list[Triple]:
    """Triples of the mentions under one relation scheme."""
    return [(m.head, mention_relation(m, scheme, base), m.tail) for m in mentions]
