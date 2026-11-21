from src.data.mkg_w_config import RelationTemplate
from src.data.mkg_w_queries import Direction, DirectionStat, DirectionStatus
from src.data.mkg_w_raw import Triple
from src.data.schemas import Split
from src.data.vn_heritage_queries import (
    DensityTier,
    article_triples,
    density_tier,
    first_label,
    split_claims,
    to_ids,
    triples_within_two_hops,
    worded_directions,
)


def test_first_label_follows_language_order() -> None:
    assert first_label({"vi": "Huế", "en": "Hue"}, ["en", "vi"]) == "Hue"
    assert first_label({"vi": "Huế"}, ["en", "vi"]) == "Huế"
    assert first_label({"fr": "Hué"}, ["en", "vi"]) is None


def test_split_claims_keeps_a_pair_together_and_structure_in_train() -> None:
    places = frozenset(f"P{i}" for i in range(200))
    claims = [(f"P{i}", "part_of", f"X{i}") for i in range(200)]
    claims += [(f"X{i}", "has_part", f"P{i}") for i in range(200)]
    claims += [("A1", "located_in_admin", "A2")]
    by_split = split_claims(claims, places, 13, 0.15, 0.25)
    split_of = {triple: split for split, triples in by_split.items() for triple in triples}
    assert all(
        split_of[(f"P{i}", "part_of", f"X{i}")] == split_of[(f"X{i}", "has_part", f"P{i}")] for i in range(200)
    )
    assert split_of[("A1", "located_in_admin", "A2")] is Split.TRAIN
    assert all(by_split[split] for split in Split)
    assert by_split == split_claims(claims, places, 13, 0.15, 0.25)


def test_article_triples_are_prefixed_and_deduplicated() -> None:
    triples = [("P1", "worships", "D1"), ("P1", "worships", "D1"), ("P1", "mentions_person", "D1")]
    assert article_triples(triples, "article:") == [
        ("P1", "article:worships", "D1"),
        ("P1", "article:mentions_person", "D1"),
    ]


def test_to_ids_maps_names() -> None:
    assert to_ids([("a", "r", "b")], {"a": 0, "b": 5}, {"r": 2}) == [Triple(0, 2, 5)]


def test_worded_directions_drop_directions_without_wording() -> None:
    stat = DirectionStat(0, Direction.HEAD, 10, 5, 0.5, DirectionStatus.KEPT)
    stats = {(0, Direction.HEAD): stat, (0, Direction.TAIL): stat, (1, Direction.HEAD): stat}
    templates = {0: RelationTemplate("r", "r", None, "a place in {anchor}")}
    assert worded_directions(stats, templates) == {(0, Direction.HEAD): stat}


def test_two_hop_count_covers_the_anchor_and_its_neighbours() -> None:
    triples = [Triple(0, 0, 1), Triple(1, 0, 2), Triple(2, 0, 3), Triple(4, 0, 5)]
    assert triples_within_two_hops([0, 4, 9], triples) == {0: 2, 4: 1, 9: 0}


def test_density_tier_bounds_are_inclusive() -> None:
    assert [density_tier(n, 10, 50) for n in (0, 10, 11, 50, 51)] == [
        DensityTier.SPARSE, DensityTier.SPARSE, DensityTier.MEDIUM, DensityTier.MEDIUM, DensityTier.DENSE,
    ]


def test_graph_variants_keep_fact_rows_aligned() -> None:
    from scripts.run_vn_heritage_retrieval import graph_variant

    triples = {"train": [[0, 3, 1], [1, 4, 2]], "articles": {"generic": [[2, 5, 0]]}}

    graph, rows = graph_variant("claims_untyped_undirected", triples, seed=13)

    assert graph == [(0, 0, 1), (1, 0, 2), (1, 0, 0), (2, 0, 1)]
    assert rows == {(0, 3, 1): [0, 2], (1, 4, 2): [1, 3]}
    assert graph_variant("generic", triples, seed=13)[0] == [(0, 3, 1), (1, 4, 2), (2, 5, 0)]
    assert graph_variant("no_edges", triples, seed=13) == ([], {})
