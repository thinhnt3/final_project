import pytest

from src.core.exceptions import DataError
from src.data.export import build_queries, select_by_id, split_of_image
from src.data.schemas import Caption, JoinedImage, KarpathySplit, Split


def _image(image_id, caption_ids):
    return JoinedImage(
        vg_image_id=image_id, coco_id=image_id * 10, url="u", width=1, height=1,
        karpathy_split=KarpathySplit.TRAIN,
        captions=tuple(Caption(c, f"text {c}") for c in caption_ids),
    )


def test_split_of_image_inverts_and_rejects_overlap():
    assert split_of_image({Split.TRAIN: [1, 2], Split.TEST: [3]}) == {
        1: Split.TRAIN, 2: Split.TRAIN, 3: Split.TEST,
    }
    with pytest.raises(DataError, match="both"):
        split_of_image({Split.TRAIN: [1], Split.VAL: [1]})


def test_select_by_id_sorts_and_requires_all():
    rows = [{"vg_image_id": 3}, {"vg_image_id": 1}, {"vg_image_id": 2}]
    assert [r["vg_image_id"] for r in select_by_id(rows, {3, 1})] == [1, 3]
    with pytest.raises(DataError, match="missing"):
        select_by_id(rows, {1, 9})


def test_build_queries_one_per_caption():
    images = [_image(1, [11, 12]), _image(2, [21])]
    queries = build_queries(images, {1: Split.TRAIN, 2: Split.TEST})
    assert [(q.query_id, q.text, q.relevant_ids, q.split) for q in queries] == [
        ("coco-11", "text 11", (1,), Split.TRAIN),
        ("coco-12", "text 12", (1,), Split.TRAIN),
        ("coco-21", "text 21", (2,), Split.TEST),
    ]


def test_build_queries_rejects_duplicate_caption_ids():
    with pytest.raises(DataError, match="duplicate"):
        build_queries([_image(1, [5]), _image(2, [5])], {1: Split.TRAIN, 2: Split.TRAIN})
