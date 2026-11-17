"""Parse Karpathy's `dataset_coco.json` into COCO images with split and captions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.exceptions import DataError
from src.data.json_io import read_json
from src.data.schemas import Caption, CocoImage, KarpathySplit


def parse_karpathy_coco(payload: dict[str, Any]) -> dict[int, CocoImage]:
    """Map coco_id → CocoImage. Captions are sorted by caption_id and kept verbatim (stripped)."""
    images = payload.get("images")
    if not isinstance(images, list):
        raise DataError("Karpathy file: 'images' must be a list")
    result: dict[int, CocoImage] = {}
    for row in images:
        coco_id = int(row["cocoid"])
        if coco_id in result:
            raise DataError(f"Karpathy file: duplicate cocoid {coco_id}")
        captions = sorted(
            (Caption(int(s["sentid"]), str(s["raw"]).strip()) for s in row["sentences"]),
            key=lambda c: c.caption_id,
        )
        result[coco_id] = CocoImage(
            coco_id=coco_id,
            karpathy_split=KarpathySplit(row["split"]),
            captions=tuple(captions),
        )
    return result


def load_karpathy_coco(path: str | Path) -> dict[int, CocoImage]:
    """Read and parse the Karpathy COCO file."""
    return parse_karpathy_coco(read_json(path))
