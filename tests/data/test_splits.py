import random

import pytest

from src.core.exceptions import DataError
from src.data.schemas import KarpathySplit, Split
from src.data.splits import SplitCandidate, make_vg_coco_split, select_candidates

SIZES = {Split.TRAIN: 350, Split.VAL: 75, Split.TEST: 75}


@pytest.fixture
def candidates():
    """1,200 images: 200 Karpathy-test, the rest spread over train/val/restval."""
    rng = random.Random(0)
    others = [KarpathySplit.TRAIN, KarpathySplit.VAL, KarpathySplit.RESTVAL]
    return [
        SplitCandidate(
            vg_image_id=i,
            karpathy_split=KarpathySplit.TEST if i < 200 else rng.choice(others),
            relation_count=rng.randint(1, 40),
        )
        for i in range(1200)
    ]


def test_sizes_and_no_overlap(candidates):
    ids = make_vg_coco_split(candidates, SIZES, n_strata=4, seed=13).ids
    assert {s: len(v) for s, v in ids.items()} == SIZES
    train, val, test = (set(ids[s]) for s in (Split.TRAIN, Split.VAL, Split.TEST))
    assert not (train & val or train & test or val & test)


def test_karpathy_test_only_in_test(candidates):
    ids = make_vg_coco_split(candidates, SIZES, n_strata=4, seed=13).ids
    karpathy_test = {c.vg_image_id for c in candidates if c.karpathy_split is KarpathySplit.TEST}
    assert set(ids[Split.TEST]) <= karpathy_test
    assert not (set(ids[Split.TRAIN]) | set(ids[Split.VAL])) & karpathy_test


def test_same_seed_same_split_regardless_of_input_order(candidates):
    first = make_vg_coco_split(candidates, SIZES, n_strata=4, seed=13)
    shuffled = candidates[:]
    random.Random(1).shuffle(shuffled)
    again = make_vg_coco_split(shuffled, SIZES, n_strata=4, seed=13)
    other = make_vg_coco_split(candidates, SIZES, n_strata=4, seed=37)
    assert first.to_dict() == again.to_dict()
    assert first.ids != other.ids


def _max_gap(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def test_each_split_follows_its_pool(candidates):
    """Stratification guarantees each split mirrors the pool it is drawn from."""
    result = make_vg_coco_split(candidates, SIZES, n_strata=4, seed=13)
    pct, pools = result.strata_pct, result.pool_strata_pct
    assert _max_gap(pct[Split.TEST], pools["karpathy_test"]) <= 2.0
    assert _max_gap(pct[Split.TRAIN], pools["karpathy_other"]) <= 2.0
    assert _max_gap(pct[Split.VAL], pools["karpathy_other"]) <= 2.0
    assert _max_gap(pct[Split.TRAIN], pct[Split.VAL]) <= 2.0


def test_too_small_test_pool_raises(candidates):
    with pytest.raises(DataError, match="Karpathy test pool has 200 images, need 500"):
        make_vg_coco_split(candidates, {**SIZES, Split.TEST: 500}, n_strata=4, seed=13)


def test_select_candidates_filters():
    karpathy = {1: KarpathySplit.TEST, 2: KarpathySplit.TRAIN, 3: KarpathySplit.TRAIN, 4: KarpathySplit.VAL}
    relations = {1: 5, 2: 0, 3: 7}  # image 4 has no graph
    selected = select_candidates(karpathy, relations, min_relations=1, exclude_ids=[3])
    assert [c.vg_image_id for c in selected] == [1]
