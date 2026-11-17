"""Train/val/test splits by image.

`make_vg_coco_split` is the Phase 2 split (scikit-learn, stratified by graph density, test
drawn only from the Karpathy test split). `make_split` / `save_split` serve the legacy
40-image subset and go away when Phase 1 moves to `vg_coco`.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.core.exceptions import DataError
from src.data.schemas import KarpathySplit, Split

# Names of the two pools splits are drawn from, used as keys in the split report.
TEST_POOL = "karpathy_test"
TRAINVAL_POOL = "karpathy_other"

@dataclass(frozen=True)
class SplitCandidate:
    """What the splitter needs to know about one image."""

    vg_image_id: int
    karpathy_split: KarpathySplit
    relation_count: int


@dataclass(frozen=True)
class SplitResult:
    """Image ids per split plus the strata used to balance them."""

    ids: dict[Split, list[int]]
    strata_edges: list[float]
    strata_pct: dict[Split, dict[int, float]]
    pool_strata_pct: dict[str, dict[int, float]]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "strata_edges": self.strata_edges,
            "strata_pct": {s.value: _str_keys(pct) for s, pct in self.strata_pct.items()},
            "pool_strata_pct": {name: _str_keys(pct) for name, pct in self.pool_strata_pct.items()},
            **{s.value: ids for s, ids in self.ids.items()},
        }


def select_candidates(
    karpathy_splits: Mapping[int, KarpathySplit],
    relation_counts: Mapping[int, int],
    *,
    min_relations: int,
    exclude_ids: Iterable[int],
) -> list[SplitCandidate]:
    """Images with a scene graph of at least `min_relations` relations, minus `exclude_ids`."""
    excluded = set(exclude_ids)
    return [
        SplitCandidate(image_id, karpathy_splits[image_id], relation_counts[image_id])
        for image_id in sorted(karpathy_splits)
        if image_id in relation_counts
        and relation_counts[image_id] >= min_relations
        and image_id not in excluded
    ]


def make_vg_coco_split(
    candidates: list[SplitCandidate], sizes: Mapping[Split, int], n_strata: int, seed: int
) -> SplitResult:
    """Draw train/val/test with scikit-learn, stratified by relation-count quantile.

    Test comes only from Karpathy-test images; train and val only from the other Karpathy
    splits, so no Karpathy-test image is ever used for training or tuning. Stratification
    keeps each split's density mix close to that of the pool it is drawn from.
    Candidates are sorted by id first, so the result does not depend on input order.
    """
    candidates = sorted(candidates, key=lambda c: c.vg_image_id)
    counts = np.array([c.relation_count for c in candidates], dtype=float)
    edges = _quantile_edges(counts, n_strata)
    stratum = {c.vg_image_id: _stratum_of(c.relation_count, edges) for c in candidates}

    test_pool = [c.vg_image_id for c in candidates if c.karpathy_split is KarpathySplit.TEST]
    trainval_pool = [c.vg_image_id for c in candidates if c.karpathy_split is not KarpathySplit.TEST]

    test = _draw(test_pool, sizes[Split.TEST], stratum, seed, pool_name="Karpathy test")
    trainval = _draw(trainval_pool, sizes[Split.TRAIN] + sizes[Split.VAL], stratum, seed, pool_name="train/val")
    train, val = train_test_split(
        trainval,
        test_size=sizes[Split.VAL],
        stratify=[stratum[i] for i in trainval],
        random_state=seed,
    )
    ids = {Split.TRAIN: sorted(train), Split.VAL: sorted(val), Split.TEST: sorted(test)}
    validate_no_overlap({s.value: v for s, v in ids.items()})
    return SplitResult(
        ids=ids,
        strata_edges=edges.tolist(),
        strata_pct={s: _strata_pct(v, stratum) for s, v in ids.items()},
        pool_strata_pct={
            TEST_POOL: _strata_pct(test_pool, stratum),
            TRAINVAL_POOL: _strata_pct(trainval_pool, stratum),
        },
    )


def _quantile_edges(counts: np.ndarray, n_strata: int) -> np.ndarray:
    """Interior bin edges at the 1/n … (n-1)/n quantiles, de-duplicated for tied counts."""
    if n_strata < 1:
        raise DataError(f"n_strata must be >= 1, got {n_strata}")
    if counts.size == 0:
        raise DataError("no candidate images to split")
    return np.unique(np.quantile(counts, np.linspace(0, 1, n_strata + 1)[1:-1]))


def _stratum_of(relation_count: int, edges: np.ndarray) -> int:
    # right=True puts a count equal to an edge in the lower bin, so with edges [9, 14, 20]
    # the bins are <=9, 10–14, 15–20, >20.
    return int(np.digitize(relation_count, edges, right=True))


def _draw(pool: list[int], size: int, stratum: Mapping[int, int], seed: int, *, pool_name: str) -> list[int]:
    if size > len(pool):
        raise DataError(f"{pool_name} pool has {len(pool)} images, need {size}")
    if size == len(pool):
        return list(pool)
    chosen, _ = train_test_split(
        pool, train_size=size, stratify=[stratum[i] for i in pool], random_state=seed
    )
    return list(chosen)


def _str_keys(pct: Mapping[int, float]) -> dict[str, float]:
    return {str(k): v for k, v in pct.items()}


def _strata_pct(ids: list[int], stratum: Mapping[int, int]) -> dict[int, float]:
    counts = Counter(stratum[i] for i in ids)
    return {k: round(100.0 * counts[k] / len(ids), 2) for k in sorted(counts)}


def validate_no_overlap(split: dict[str, list[int]]) -> None:
    """Raise when any image id appears in more than one split."""
    sets = {name: set(ids) for name, ids in split.items() if name in {"train", "val", "test"}}
    for left in sets:
        for right in sets:
            if left < right and sets[left] & sets[right]:
                raise DataError(f"split overlap: {left}/{right}")


def make_split(image_ids: list[int], seed: int = 13, train_ratio: float = 0.5, val_ratio: float = 0.25) -> dict[str, list[int]]:
    """Legacy random split for the 40-image subset."""
    ids = list(dict.fromkeys(map(int, image_ids)))
    random.Random(seed).shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    n_val = max(1, int(len(ids) * val_ratio))
    return {"train": sorted(ids[:n_train]), "val": sorted(ids[n_train:n_train + n_val]), "test": sorted(ids[n_train + n_val:])}


def save_split(split: dict[str, list[int]], path: str | Path, seed: int) -> None:
    """Legacy writer for the 40-image subset."""
    payload = {"seed": seed, **split}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
