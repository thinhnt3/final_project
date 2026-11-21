from src.data.schemas import Split
from src.data.vn_heritage_core import (
    CoreRules,
    admin_hierarchy,
    assign_split,
    century_node,
    class_like_places,
    drop_shadowed,
    inception_century,
    item_triples,
    select_core,
    split_groups,
)

PROVINCE = frozenset({"province"})


def _rules(min_triples: int = 2, min_images: int = 2) -> CoreRules:
    return CoreRules(
        min_triples=min_triples,
        min_images=min_images,
        label_language="vi",
        class_relations=frozenset({"instance_of"}),
        shadowed_relations={"located_in_province": "located_in_admin"},
        admin_relations=frozenset({"located_in_admin"}),
        hierarchy_relation="located_in_admin",
        group_relations=frozenset({"part_of"}),
        province_classes=PROVINCE,
    )


def test_class_like_places_are_tails_of_class_relations_only() -> None:
    triples = [("P1", "instance_of", "P2"), ("P1", "part_of", "P3"), ("P1", "instance_of", "C1")]
    assert class_like_places({"P1", "P2", "P3"}, triples, frozenset({"instance_of"})) == {"P2"}


def test_drop_shadowed_removes_only_identical_pairs() -> None:
    triples = [
        ("P1", "located_in_admin", "HCM"),
        ("P1", "located_in_province", "HCM"),
        ("P2", "located_in_admin", "Q1"),
        ("P2", "located_in_province", "HCM"),
    ]
    kept = drop_shadowed(triples, {"located_in_province": "located_in_admin"})
    assert ("P1", "located_in_province", "HCM") not in kept
    assert ("P2", "located_in_province", "HCM") in kept
    assert len(kept) == 3


def test_select_core_applies_every_condition() -> None:
    labels = {
        "ok": {"vi": "Chùa A"},
        "few_triples": {"vi": "Chùa B"},
        "few_images": {"vi": "Chùa C"},
        "no_vi": {"en": "Temple D"},
        "category": {"vi": "chùa Việt Nam"},
    }
    triples = [(q, "religion", f"t{i}") for q in ("ok", "few_images", "no_vi", "category") for i in range(2)]
    triples += [("few_triples", "religion", "t0"), ("ok", "instance_of", "category")]
    images = {"ok": 2, "few_triples": 5, "few_images": 1, "no_vi": 5, "category": 5}
    assert select_core(labels, triples, images, _rules()) == ["ok"]


def test_admin_hierarchy_stops_at_province_and_skips_unlabelled() -> None:
    parents = {"ward": ["district"], "district": ["province", "ghost"], "province": ["country"]}
    classes = {"province": ["province"]}
    edges, skipped = admin_hierarchy(
        ["ward"], parents, classes, PROVINCE, {"ward", "district", "province", "country"}, "located_in_admin"
    )
    assert edges == [("district", "located_in_admin", "province"), ("ward", "located_in_admin", "district")]
    assert skipped == 1


def test_admin_hierarchy_terminates_on_cycles() -> None:
    edges, _ = admin_hierarchy(["a"], {"a": ["b"], "b": ["a"]}, {}, PROVINCE, {"a", "b"}, "in")
    assert edges == [("a", "in", "b"), ("b", "in", "a")]


def test_split_groups_join_direct_and_shared_target() -> None:
    triples = [("P1", "part_of", "P2"), ("P3", "part_of", "X"), ("P4", "part_of", "X"), ("P5", "religion", "X")]
    groups = split_groups(["P1", "P2", "P3", "P4", "P5"], triples, frozenset({"part_of"}))
    assert groups["P1"] == groups["P2"] == "P1"
    assert groups["P3"] == groups["P4"] == "P3"
    assert groups["P5"] == "P5"


def test_assign_split_is_deterministic_and_covers_all_splits() -> None:
    splits = [assign_split(f"Q{i}", 13, 0.15, 0.15) for i in range(400)]
    assert splits == [assign_split(f"Q{i}", 13, 0.15, 0.15) for i in range(400)]
    assert set(splits) == {Split.TRAIN, Split.VAL, Split.TEST}
    assert 0.6 < splits.count(Split.TRAIN) / 400 < 0.8


def _claim(value: object, rank: str = "normal", snaktype: str = "value") -> dict[str, object]:
    return {"rank": rank, "mainsnak": {"snaktype": snaktype, "datavalue": {"value": value}}}


def test_item_triples_skip_deprecated_missing_and_non_item_values() -> None:
    entities = {
        "P1": {"claims": {
            "P177": [_claim({"id": "River"}), _claim({"id": "Old"}, rank="deprecated"), _claim({"id": "River"})],
            "P84": [_claim({"id": "X"}, snaktype="somevalue"), _claim("free text")],
            "P999": [_claim({"id": "Ignored"})],
        }},
        "P2": {},
    }
    assert item_triples(entities, {"P177": "crosses", "P84": "architect"}) == [("P1", "crosses", "River")]


def _date(time: str, precision: int) -> dict[str, object]:
    return _claim({"time": time, "precision": precision})


def test_inception_century_handles_precision_and_era() -> None:
    assert inception_century({"claims": {"P571": [_date("+1802-00-00T00:00:00Z", 9)]}}, "P571") == 19
    assert inception_century({"claims": {"P571": [_date("+1900-00-00T00:00:00Z", 7)]}}, "P571") == 19
    assert inception_century({"claims": {"P571": [_date("+2000-01-01T00:00:00Z", 11)]}}, "P571") == 20
    assert inception_century({"claims": {"P571": [_date("+1000-00-00T00:00:00Z", 6)]}}, "P571") is None
    assert inception_century({"claims": {"P571": [_date("-0200-00-00T00:00:00Z", 9)]}}, "P571") is None
    assert inception_century({}, "P571") is None


def test_century_node_is_shared_by_places_of_the_same_century() -> None:
    assert century_node(19) == ("century:19", {"vi": "thế kỷ 19", "en": "century 19"})
