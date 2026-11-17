import pytest

from src.core.exceptions import DataError
from src.data.coco_captions import parse_karpathy_coco
from src.data.schemas import JoinedImage, KarpathySplit
from src.data.vg_coco_join import join_vg_coco


def _karpathy_image(cocoid, split, sentids):
    return {
        "cocoid": cocoid,
        "split": split,
        "sentences": [{"sentid": s, "raw": f"  caption {s}  "} for s in sentids],
    }


@pytest.fixture
def coco():
    return parse_karpathy_coco({"images": [
        _karpathy_image(100, "train", [5, 3, 1, 4, 2]),         # unsorted → sorted
        _karpathy_image(200, "test", [10, 11, 12, 13, 14, 15]),  # 6 captions → keep 5 smallest
        _karpathy_image(300, "restval", [20, 21, 22]),           # too few
        _karpathy_image(400, "val", [30, 31, 32, 33, 34]),       # shared by two VG images
    ]})


def _vg(image_id, coco_id):
    return {"image_id": image_id, "coco_id": coco_id, "url": f"u/{image_id}.jpg", "width": 10, "height": 20}


@pytest.fixture
def vg_images():
    return [
        _vg(2, 200),
        _vg(1, 100),
        _vg(3, 300),
        _vg(4, 400),
        _vg(5, 400),
        _vg(6, None),
        _vg(7, 999),  # not in Karpathy
    ]


def test_parse_sorts_and_strips_captions(coco):
    captions = coco[100].captions
    assert [c.caption_id for c in captions] == [1, 2, 3, 4, 5]
    assert captions[0].text == "caption 1"
    assert coco[200].karpathy_split is KarpathySplit.TEST


def test_parse_rejects_duplicate_cocoid():
    image = _karpathy_image(1, "train", [1])
    with pytest.raises(DataError, match="duplicate"):
        parse_karpathy_coco({"images": [image, image]})


def test_join_keeps_only_valid_images(coco, vg_images):
    joined, _ = join_vg_coco(vg_images, coco, captions_per_image=5)
    assert [j.vg_image_id for j in joined] == [1, 2]
    assert [c.caption_id for c in joined[1].captions] == [10, 11, 12, 13, 14]


def test_funnel_accounts_for_every_drop(coco, vg_images):
    _, funnel = join_vg_coco(vg_images, coco, captions_per_image=5)
    stages = funnel.to_dict()["stages"]
    assert [(s["stage"], s["kept"], s["dropped"]) for s in stages] == [
        ("vg_images", 7, 0),
        ("has_coco_id", 6, 1),
        ("unique_coco_id", 4, 2),
        ("in_karpathy", 3, 1),
        ("enough_captions", 2, 1),
    ]


def test_joined_image_roundtrip(coco, vg_images):
    joined, _ = join_vg_coco(vg_images, coco, captions_per_image=5)
    assert JoinedImage.from_dict(joined[0].to_dict()) == joined[0]
