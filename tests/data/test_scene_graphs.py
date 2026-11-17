import pytest

from src.data.scene_graphs import (
    NormalizationReport,
    build_synset_backfill,
    normalize_scene_graph,
    normalize_scene_graphs,
)
from src.data.schemas import SceneGraph, SynsetSource


def _obj(object_id, name, synsets):
    return {"object_id": object_id, "names": [name] if name else [], "synsets": synsets}


def _rel(subject_id, object_id, predicate):
    return {"subject_id": subject_id, "object_id": object_id, "predicate": predicate, "synsets": ["x.r.01"]}


@pytest.fixture
def record():
    return {
        "image_id": 1,
        "objects": [
            _obj(10, "Bikes", ["bicycle.n.01"]),
            _obj(11, "guy", ["guy.n.01"]),
            _obj(12, "Tree Trunk", []),                      # no synset, not in backfill
            _obj(13, "", ["man.n.01"]),                      # no name → dropped
            _obj(10, "bike", ["bicycle.n.01"]),              # duplicate id → dropped
            _obj(14, "sneakers", ["gym_shoe.n.01", "shoe.n.01"]),
            _obj(15, "leaves", []),                          # no synset, in backfill
        ],
        "relationships": [
            _rel(11, 10, "RIDING"),
            _rel(11, 10, "riding a"),  # duplicate after normalization
            _rel(11, 13, "on"),        # object 13 dropped → dangling
            _rel(11, 11, "is"),        # self loop
            _rel(11, 12, "..."),       # empty predicate
            _rel(11, 14, "are wearing"),
        ],
    }


BACKFILL = {"leaves": "leaf.n.01"}


def test_objects_get_canonical_labels(record):
    graph = normalize_scene_graph(record, BACKFILL, NormalizationReport())
    labels = {o.object_id: (o.label, o.synset, o.synset_source) for o in graph.objects}
    assert labels == {
        10: ("bicycle", "bicycle.n.01", SynsetSource.VG),
        11: ("guy", "guy.n.01", SynsetSource.VG),            # synsets do not merge near-synonyms
        12: ("tree trunk", None, SynsetSource.NONE),
        14: ("gym shoe", "gym_shoe.n.01", SynsetSource.VG),
        15: ("leaf", "leaf.n.01", SynsetSource.BACKFILL),
    }


def test_relations_cleaned_and_counted(record):
    report = NormalizationReport()
    graph = normalize_scene_graph(record, BACKFILL, report)
    assert [(r.subject_id, r.predicate, r.object_id) for r in graph.relations] == [
        (11, "riding", 10),
        (11, "wearing", 14),
    ]
    assert graph.relations[0].predicate_raw == "RIDING"
    assert dict(report.dropped) == {
        "object_no_name": 1,
        "object_duplicate_id": 1,
        "relation_duplicate": 1,
        "relation_dangling": 1,
        "relation_self_loop": 1,
        "relation_empty_predicate": 1,
    }


def test_report_counts(record):
    report = NormalizationReport()
    normalize_scene_graph(record, BACKFILL, report)
    summary = report.to_dict()
    assert summary["objects"] == {
        "kept": 5,
        "by_synset_source": {"vg": 3, "backfill": 1, "none": 1},
        "with_synset_pct": 80.0,
        "multi_synset": 1,
    }
    assert summary["top_unmapped_names"] == [("tree trunk", 1)]
    assert summary["relations_kept"] == 2


def _named(name, synset, n):
    return [_obj(i, name, [synset] if synset else []) for i in range(n)]


def test_backfill_requires_count_and_share():
    records = [{"image_id": 1, "objects": [
        *_named("leaves", "leaf.n.01", 9), *_named("leaves", "tree.n.01", 1),     # 90% → kept
        *_named("orange", "orange.n.01", 3), *_named("orange", "vest.n.01", 3),  # split → rejected
        *_named("top", "shirt.n.01", 4),                                         # too few → rejected
        *_named("Clouds", "cloud.n.01", 5),                                      # name is normalized
        *_named("leaves", None, 50),                                             # unannotated: no vote
    ]}]
    assert build_synset_backfill(records, min_count=5, min_share=0.9) == {
        "leaves": "leaf.n.01",
        "clouds": "cloud.n.01",
    }


def test_filters_by_image_id_and_counts_missing(record):
    other = {**record, "image_id": 2}
    graphs, report = normalize_scene_graphs(
        [other, record], image_ids={1, 3}, backfill_min_count=1, backfill_min_share=0.9
    )
    assert [g.vg_image_id for g in graphs] == [1]
    assert report.images_missing_graph == 1


def test_image_without_relations_is_counted():
    report = NormalizationReport()
    normalize_scene_graph({"image_id": 5, "objects": [_obj(1, "sky", ["sky.n.01"])], "relationships": []}, {}, report)
    assert report.images_without_relations == 1


def test_scene_graph_roundtrip(record):
    graph = normalize_scene_graph(record, BACKFILL, NormalizationReport())
    assert SceneGraph.from_dict(graph.to_dict()) == graph
