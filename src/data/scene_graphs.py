"""Normalize raw Visual Genome scene graphs: one canonical label per object, cleaned predicates.

Rules (see Phase 2, step 2.3):
- object label = lemma of its first WordNet synset;
- objects without a synset borrow one from other objects with the same raw name when that
  name is frequent and unambiguous (backfill); the rest keep their normalized raw name;
- predicates are normalized as text only, because VG predicate synsets lose meaning
  (`ON` → `along.r.01`);
- relations are dropped when dangling, self-looping, empty after normalization, or a
  duplicate (subject label, predicate, object label) within the same image.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.data.schemas import Relation, SceneGraph, SceneObject, SynsetSource
from src.data.synsets import normalize_predicate, normalize_text, synset_lemma


class DropReason(StrEnum):
    """Why an object or relation was removed."""

    OBJECT_NO_NAME = "object_no_name"
    OBJECT_DUPLICATE_ID = "object_duplicate_id"
    RELATION_DANGLING = "relation_dangling"
    RELATION_SELF_LOOP = "relation_self_loop"
    RELATION_EMPTY_PREDICATE = "relation_empty_predicate"
    RELATION_DUPLICATE = "relation_duplicate"


@dataclass
class NormalizationReport:
    """Counters accumulated over all normalized images."""

    images: int = 0
    images_missing_graph: int = 0
    images_without_relations: int = 0
    backfill_names: int = 0
    objects_kept: int = 0
    objects_by_source: Counter[str] = field(default_factory=Counter)
    objects_multi_synset: int = 0
    relations_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    raw_names: Counter[str] = field(default_factory=Counter)
    labels: Counter[str] = field(default_factory=Counter)
    synsets: Counter[str] = field(default_factory=Counter)
    unmapped_names: Counter[str] = field(default_factory=Counter)
    raw_predicates: Counter[str] = field(default_factory=Counter)
    predicates: Counter[str] = field(default_factory=Counter)

    def to_dict(self, top_k: int = 50) -> dict[str, Any]:
        """JSON-ready summary; long counters are cut to their `top_k` most common."""
        with_synset = self.objects_by_source[SynsetSource.VG] + self.objects_by_source[SynsetSource.BACKFILL]
        return {
            "images": self.images,
            "images_missing_graph": self.images_missing_graph,
            "images_without_relations": self.images_without_relations,
            "objects": {
                "kept": self.objects_kept,
                "by_synset_source": {str(s): self.objects_by_source[s] for s in SynsetSource},
                "with_synset_pct": _pct(with_synset, self.objects_kept),
                "multi_synset": self.objects_multi_synset,
            },
            "backfill_names": self.backfill_names,
            "relations_kept": self.relations_kept,
            "dropped": {str(reason): count for reason, count in sorted(self.dropped.items())},
            "vocabulary": {
                "distinct_raw_names": len(self.raw_names),
                "distinct_synsets": len(self.synsets),
                "distinct_labels": len(self.labels),
                "distinct_raw_predicates": len(self.raw_predicates),
                "distinct_predicates": len(self.predicates),
            },
            "top_labels": self.labels.most_common(top_k),
            "top_predicates": self.predicates.most_common(top_k),
            "top_unmapped_names": self.unmapped_names.most_common(top_k),
        }


def build_synset_backfill(
    records: Iterable[dict[str, Any]], min_count: int, min_share: float
) -> dict[str, str]:
    """Map raw name → synset for names whose annotated objects agree on one synset.

    A name qualifies when it carries a synset at least `min_count` times and its most
    common synset covers at least `min_share` of those, which keeps ambiguous names
    such as `orange` (colour or fruit) out.
    """
    votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for raw in record.get("objects", []):
            name = _object_name(raw)
            synsets = raw.get("synsets") or []
            if name and synsets:
                votes[name][synsets[0]] += 1
    backfill: dict[str, str] = {}
    for name, counter in votes.items():
        total = sum(counter.values())
        synset, top = counter.most_common(1)[0]
        if total >= min_count and top / total >= min_share:
            backfill[name] = synset
    return backfill


def normalize_scene_graph(
    record: dict[str, Any], backfill: Mapping[str, str], report: NormalizationReport
) -> SceneGraph:
    """Normalize one raw VG record, adding its counts to `report`."""
    objects: dict[int, SceneObject] = {}
    for raw in record.get("objects", []):
        name_raw = _object_name(raw)
        if not name_raw:
            report.dropped[DropReason.OBJECT_NO_NAME] += 1
            continue
        object_id = int(raw["object_id"])
        if object_id in objects:
            report.dropped[DropReason.OBJECT_DUPLICATE_ID] += 1
            continue
        obj = _make_object(object_id, name_raw, raw.get("synsets") or [], backfill)
        objects[object_id] = obj
        _count_object(report, obj, multi_synset=len(raw.get("synsets") or []) > 1)

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in record.get("relationships", []):
        subject_id, object_id = int(raw["subject_id"]), int(raw["object_id"])
        predicate_raw = str(raw.get("predicate") or "")
        predicate = normalize_predicate(predicate_raw)
        reason = _relation_drop_reason(subject_id, object_id, predicate, objects)
        if reason is None:
            key = (objects[subject_id].label, predicate, objects[object_id].label)
            if key in seen:
                reason = DropReason.RELATION_DUPLICATE
            seen.add(key)
        if reason is not None:
            report.dropped[reason] += 1
            continue
        relations.append(Relation(subject_id, object_id, predicate_raw, predicate))
        report.raw_predicates[predicate_raw] += 1
        report.predicates[predicate] += 1

    report.images += 1
    report.relations_kept += len(relations)
    report.images_without_relations += not relations
    return SceneGraph(int(record["image_id"]), tuple(objects.values()), tuple(relations))


def normalize_scene_graphs(
    records: list[dict[str, Any]], image_ids: set[int], *, backfill_min_count: int, backfill_min_share: float
) -> tuple[list[SceneGraph], NormalizationReport]:
    """Normalize the records whose image id is in `image_ids`, sorted by image id.

    The backfill table is learned from the same selected records, so it reflects the
    vocabulary of the images we actually use.
    """
    selected = [r for r in records if int(r["image_id"]) in image_ids]
    backfill = build_synset_backfill(selected, backfill_min_count, backfill_min_share)
    report = NormalizationReport(backfill_names=len(backfill))
    graphs = [normalize_scene_graph(record, backfill, report) for record in selected]
    report.images_missing_graph = len(image_ids) - len(graphs)
    graphs.sort(key=lambda g: g.vg_image_id)
    return graphs, report


def _object_name(raw: dict[str, Any]) -> str:
    names = raw.get("names") or ([raw["name"]] if raw.get("name") else [])
    return normalize_text(names[0]) if names else ""


def _make_object(object_id: int, name_raw: str, synsets: list[str], backfill: Mapping[str, str]) -> SceneObject:
    if synsets:
        return SceneObject(object_id, name_raw, synsets[0], SynsetSource.VG, synset_lemma(synsets[0]))
    if name_raw in backfill:
        synset = backfill[name_raw]
        return SceneObject(object_id, name_raw, synset, SynsetSource.BACKFILL, synset_lemma(synset))
    return SceneObject(object_id, name_raw, None, SynsetSource.NONE, name_raw)


def _count_object(report: NormalizationReport, obj: SceneObject, *, multi_synset: bool) -> None:
    report.objects_kept += 1
    report.objects_by_source[obj.synset_source] += 1
    report.objects_multi_synset += multi_synset
    report.raw_names[obj.name_raw] += 1
    report.labels[obj.label] += 1
    if obj.synset:
        report.synsets[obj.synset] += 1
    else:
        report.unmapped_names[obj.name_raw] += 1


def _relation_drop_reason(
    subject_id: int, object_id: int, predicate: str, objects: dict[int, SceneObject]
) -> DropReason | None:
    if subject_id not in objects or object_id not in objects:
        return DropReason.RELATION_DANGLING
    if subject_id == object_id:
        return DropReason.RELATION_SELF_LOOP
    if not predicate:
        return DropReason.RELATION_EMPTY_PREDICATE
    return None


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0
