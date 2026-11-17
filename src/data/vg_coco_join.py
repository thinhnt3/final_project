"""Keep Visual Genome images that are COCO images and attach their COCO captions.

Filters run in a fixed order and each one records how many images it removed, so the
whole reduction from 108K VG images is accounted for in the funnel.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.data.schemas import CocoImage, JoinedImage


class FunnelStage(StrEnum):
    """Filter stages, in the order they are applied."""

    VG_IMAGES = "vg_images"
    HAS_COCO_ID = "has_coco_id"
    UNIQUE_COCO_ID = "unique_coco_id"
    IN_KARPATHY = "in_karpathy"
    ENOUGH_CAPTIONS = "enough_captions"


@dataclass
class Funnel:
    """Image count after each stage."""

    counts: dict[FunnelStage, int] = field(default_factory=dict)

    def record(self, stage: FunnelStage, count: int) -> None:
        """Store the count that survived `stage`."""
        self.counts[stage] = count

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form with the number dropped at each stage."""
        rows = []
        previous: int | None = None
        for stage, count in self.counts.items():
            rows.append({
                "stage": stage.value,
                "kept": count,
                "dropped": 0 if previous is None else previous - count,
            })
            previous = count
        return {"stages": rows}


def join_vg_coco(
    vg_images: list[dict[str, Any]],
    coco_images: dict[int, CocoImage],
    captions_per_image: int,
) -> tuple[list[JoinedImage], Funnel]:
    """Return VG images that map to exactly one COCO image with enough captions.

    - Images whose coco_id is shared by several VG images are all dropped, so one
      photo can never land in two splits.
    - Images with more captions than needed keep the ones with the smallest caption_id.
    """
    funnel = Funnel()
    funnel.record(FunnelStage.VG_IMAGES, len(vg_images))

    with_coco = [row for row in vg_images if row.get("coco_id")]
    funnel.record(FunnelStage.HAS_COCO_ID, len(with_coco))

    per_coco = Counter(int(row["coco_id"]) for row in with_coco)
    unique = [row for row in with_coco if per_coco[int(row["coco_id"])] == 1]
    funnel.record(FunnelStage.UNIQUE_COCO_ID, len(unique))

    in_karpathy = [row for row in unique if int(row["coco_id"]) in coco_images]
    funnel.record(FunnelStage.IN_KARPATHY, len(in_karpathy))

    joined: list[JoinedImage] = []
    for row in in_karpathy:
        coco = coco_images[int(row["coco_id"])]
        if len(coco.captions) < captions_per_image:
            continue
        joined.append(JoinedImage(
            vg_image_id=int(row["image_id"]),
            coco_id=coco.coco_id,
            url=str(row["url"]),
            width=int(row["width"]),
            height=int(row["height"]),
            karpathy_split=coco.karpathy_split,
            captions=coco.captions[:captions_per_image],
        ))
    funnel.record(FunnelStage.ENOUGH_CAPTIONS, len(joined))

    joined.sort(key=lambda image: image.vg_image_id)
    return joined, funnel


def split_counts(images: list[JoinedImage]) -> dict[str, int]:
    """Number of joined images per Karpathy split."""
    counts = Counter(image.karpathy_split.value for image in images)
    return dict(sorted(counts.items()))
