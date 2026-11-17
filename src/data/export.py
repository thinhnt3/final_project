"""Select the split images from the interim data and turn their captions into queries."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.core.exceptions import DataError
from src.data.schemas import JoinedImage, Query, Split


def split_of_image(split_ids: Mapping[Split, Iterable[int]]) -> dict[int, Split]:
    """Invert {split: ids} into {image id: split}; raise if an id is in two splits."""
    result: dict[int, Split] = {}
    for split, ids in split_ids.items():
        for image_id in ids:
            if image_id in result:
                raise DataError(f"image {image_id} is in both {result[image_id]} and {split}")
            result[image_id] = split
    return result


def select_by_id(rows: Iterable[dict[str, Any]], image_ids: set[int]) -> list[dict[str, Any]]:
    """Rows whose `vg_image_id` is in `image_ids`, sorted by id; raise if any id is missing."""
    selected = sorted((r for r in rows if int(r["vg_image_id"]) in image_ids), key=lambda r: int(r["vg_image_id"]))
    missing = image_ids - {int(r["vg_image_id"]) for r in selected}
    if missing:
        raise DataError(f"{len(missing)} split images missing from input, e.g. {sorted(missing)[:5]}")
    return selected


def build_queries(images: Iterable[JoinedImage], split_by_image: Mapping[int, Split]) -> list[Query]:
    """One query per caption, id `coco-<caption_id>`, relevant only to its own image."""
    queries = [
        Query(
            query_id=f"coco-{caption.caption_id}",
            text=caption.text,
            relevant_ids=(image.vg_image_id,),
            split=split_by_image[image.vg_image_id],
        )
        for image in images
        for caption in image.captions
    ]
    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        raise DataError("duplicate query ids")
    return queries
