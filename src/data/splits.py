from __future__ import annotations

import json
import random
from pathlib import Path


def make_split(image_ids: list[int], seed: int = 13, train_ratio: float = 0.5, val_ratio: float = 0.25) -> dict[str, list[int]]:
    ids = list(dict.fromkeys(map(int, image_ids)))
    random.Random(seed).shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    n_val = max(1, int(len(ids) * val_ratio))
    return {"train": sorted(ids[:n_train]), "val": sorted(ids[n_train:n_train + n_val]), "test": sorted(ids[n_train + n_val:])}


def save_split(split: dict[str, list[int]], path: str | Path, seed: int) -> None:
    payload = {"seed": seed, **split}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def validate_no_overlap(split: dict[str, list[int]]) -> None:
    sets = {name: set(ids) for name, ids in split.items() if name in {"train", "val", "test"}}
    for left in sets:
        for right in sets:
            if left < right and sets[left] & sets[right]:
                raise ValueError(f"split overlap: {left}/{right}")
