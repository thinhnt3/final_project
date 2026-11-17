"""Record types written to and read from `data/interim` and `data/processed`."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class KarpathySplit(StrEnum):
    """Split labels used in Karpathy's `dataset_coco.json`."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    RESTVAL = "restval"


class Split(StrEnum):
    """Experiment splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Caption:
    """One COCO caption; `caption_id` is Karpathy's `sentid`."""

    caption_id: int
    text: str


@dataclass(frozen=True)
class CocoImage:
    """A COCO image as listed in the Karpathy file."""

    coco_id: int
    karpathy_split: KarpathySplit
    captions: tuple[Caption, ...]


@dataclass(frozen=True)
class JoinedImage:
    """A Visual Genome image that is also a COCO image, with its captions."""

    vg_image_id: int
    coco_id: int
    url: str
    width: int
    height: int
    karpathy_split: KarpathySplit
    captions: tuple[Caption, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "vg_image_id": self.vg_image_id,
            "coco_id": self.coco_id,
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "karpathy_split": self.karpathy_split.value,
            "captions": [{"caption_id": c.caption_id, "text": c.text} for c in self.captions],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JoinedImage:
        """Inverse of `to_dict`."""
        return cls(
            vg_image_id=int(raw["vg_image_id"]),
            coco_id=int(raw["coco_id"]),
            url=str(raw["url"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            karpathy_split=KarpathySplit(raw["karpathy_split"]),
            captions=tuple(Caption(int(c["caption_id"]), str(c["text"])) for c in raw["captions"]),
        )


class SynsetSource(StrEnum):
    """Where an object's synset came from."""

    VG = "vg"              # annotated on this object in Visual Genome
    BACKFILL = "backfill"  # borrowed from other objects with the same raw name
    NONE = "none"          # no synset; label is the normalized raw name


@dataclass(frozen=True)
class Query:
    """A text query: one COCO caption whose only relevant image is the one it describes."""

    query_id: str
    text: str
    relevant_ids: tuple[int, ...]
    split: Split

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant_ids": list(self.relevant_ids),
            "split": self.split.value,
        }


@dataclass(frozen=True)
class SceneObject:
    """A VG object with its canonical label; `synset` is None when no source gives one."""

    object_id: int
    name_raw: str
    synset: str | None
    synset_source: SynsetSource
    label: str


@dataclass(frozen=True)
class Relation:
    """A directed edge subject → object with raw and normalized predicate."""

    subject_id: int
    object_id: int
    predicate_raw: str
    predicate: str


@dataclass(frozen=True)
class SceneGraph:
    """Normalized scene graph of one VG image."""

    vg_image_id: int
    objects: tuple[SceneObject, ...]
    relations: tuple[Relation, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "vg_image_id": self.vg_image_id,
            "objects": [{**vars(o), "synset_source": o.synset_source.value} for o in self.objects],
            "relations": [vars(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SceneGraph:
        """Inverse of `to_dict`."""
        return cls(
            vg_image_id=int(raw["vg_image_id"]),
            objects=tuple(
                SceneObject(**{**o, "synset_source": SynsetSource(o["synset_source"])})
                for o in raw["objects"]
            ),
            relations=tuple(Relation(**r) for r in raw["relations"]),
        )


@dataclass(frozen=True)
class Concept:
    """A graph concept node. `key` is the synset id, or `name:<raw name>` when the objects
    have no synset. `text` is the raw name used in the most train images, for display and
    embedding."""

    concept_id: int
    key: str
    text: str
    synset: str | None
    train_images: int
    idf: float


@dataclass(frozen=True)
class Predicate:
    """A relation label kept in the graph vocabulary."""

    predicate_id: int
    text: str
    train_images: int


@dataclass(frozen=True)
class ImageGraph:
    """One image's graph over the fixed vocabulary: one node per concept, edges as id triplets."""

    vg_image_id: int
    split: Split
    concept_ids: tuple[int, ...]
    concept_counts: tuple[int, ...]  # objects of each concept in the image, aligned with concept_ids
    triplets: tuple[tuple[int, int, int], ...]  # (subject concept, predicate, object concept)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "vg_image_id": self.vg_image_id,
            "split": self.split.value,
            "concept_ids": list(self.concept_ids),
            "concept_counts": list(self.concept_counts),
            "triplets": [list(t) for t in self.triplets],
        }
