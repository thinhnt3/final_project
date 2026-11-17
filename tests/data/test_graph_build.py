import math

import pytest

from src.core.exceptions import DataError
from src.data.graph_build import (
    OBJECT_DROP_REASONS,
    GraphDropReason,
    build_graphs,
    build_image_graph,
    build_vocabulary,
    concept_key,
    graph_statistics,
)
from src.data.schemas import Relation, SceneGraph, SceneObject, Split, SynsetSource

JUNK = frozenset({"the", "white", "he", "side"})


def _obj(object_id, name, synset=None):
    source = SynsetSource.VG if synset else SynsetSource.NONE
    return SceneObject(object_id, name, synset, source, name)


def _rel(subject_id, object_id, predicate="on"):
    return Relation(subject_id, object_id, predicate, predicate)


def _graph(image_id, objects, relations=()):
    return SceneGraph(image_id, tuple(objects), tuple(relations))


def _man_on_couch(image_id, couch_name="couch", man_name="man"):
    """man —on→ couch, plus a junk object."""
    return _graph(
        image_id,
        [_obj(1, man_name, "man.n.01"), _obj(2, couch_name, "sofa.n.01"), _obj(3, "the")],
        [_rel(1, 2)],
    )


def _vocab(train, min_train_images=1):
    return build_vocabulary(train, min_train_images=min_train_images, junk_names=JUNK)


def test_concept_key_uses_synset_else_raw_name():
    assert concept_key(_obj(1, "pants", "trouser.n.01")) == "trouser.n.01"
    assert concept_key(_obj(1, "street light")) == "name:street light"


def test_vocabulary_keeps_concepts_frequent_in_train_and_picks_common_name():
    train = [_man_on_couch(1), _man_on_couch(2), _man_on_couch(3, couch_name="sofa"),
             _graph(4, [_obj(1, "zebra", "zebra.n.01")])]
    vocab = _vocab(train, min_train_images=2)

    by_key = {c.key: c for c in vocab.concepts}
    assert set(by_key) == {"man.n.01", "sofa.n.01"}  # zebra: 1 image; `the`: junk
    assert by_key["sofa.n.01"].text == "couch"  # 2 images say couch, 1 says sofa
    assert by_key["sofa.n.01"].synset == "sofa.n.01"
    assert by_key["man.n.01"].train_images == 3
    assert by_key["man.n.01"].idf == pytest.approx(math.log(4 / 3), abs=1e-6)
    assert [p.text for p in vocab.predicates] == ["on"]
    assert vocab.train_images == 4


def test_threshold_is_inclusive():
    train = [_man_on_couch(1), _man_on_couch(2)]
    assert len(_vocab(train, min_train_images=2).concepts) == 2
    assert _vocab(train, min_train_images=3).concepts == ()


def test_frequency_counts_images_not_objects_or_relations():
    # One image holding three men and three `on` relations still counts once.
    crowd = _graph(
        1,
        [_obj(1, "man", "man.n.01"), _obj(2, "man", "man.n.01"), _obj(3, "man", "man.n.01"),
         _obj(4, "couch", "sofa.n.01")],
        [_rel(1, 4), _rel(2, 4), _rel(3, 4)],
    )
    vocab = _vocab([crowd, _man_on_couch(2)], min_train_images=1)
    assert {c.key: c.train_images for c in vocab.concepts} == {"man.n.01": 2, "sofa.n.01": 2}
    assert [(p.text, p.train_images) for p in vocab.predicates] == [("on", 2)]
    # With object counts the crowd alone would pass a threshold of 3.
    assert _vocab([crowd], min_train_images=3).concepts == ()
    assert _vocab([crowd], min_train_images=2).predicates == ()


def test_display_name_counts_images_not_objects():
    # `sofa` is written three times in one image, `couch` once in each of two images.
    train = [
        _graph(1, [_obj(1, "sofa", "sofa.n.01"), _obj(2, "sofa", "sofa.n.01"), _obj(3, "sofa", "sofa.n.01")]),
        _graph(2, [_obj(1, "couch", "sofa.n.01")]),
        _graph(3, [_obj(1, "couch", "sofa.n.01")]),
    ]
    assert _vocab(train).concepts[0].text == "couch"


def test_ids_are_dense_ordered_by_train_images_then_key():
    train = [
        _graph(1, [_obj(1, "sky", "sky.n.01"), _obj(2, "dog", "dog.n.01"), _obj(3, "cat", "cat.n.01")],
               [_rel(2, 1, "under"), _rel(3, 1, "below")]),
        _graph(2, [_obj(1, "sky", "sky.n.01"), _obj(2, "dog", "dog.n.01"), _obj(3, "cat", "cat.n.01")],
               [_rel(2, 1, "under"), _rel(3, 1, "below"), _rel(2, 3, "near")]),
        _graph(3, [_obj(1, "sky", "sky.n.01"), _obj(2, "dog", "dog.n.01")], [_rel(2, 1, "near")]),
        _graph(4, [_obj(1, "sky", "sky.n.01"), _obj(3, "cat", "cat.n.01")], [_rel(3, 1, "near")]),
    ]
    vocab = _vocab(train)
    # sky in 4 images; cat and dog tie at 3 and are ordered by key.
    assert [(c.concept_id, c.key) for c in vocab.concepts] == [(0, "sky.n.01"), (1, "cat.n.01"), (2, "dog.n.01")]
    # near in 3 images; below and under tie at 2.
    assert [(p.predicate_id, p.text) for p in vocab.predicates] == [(0, "near"), (1, "below"), (2, "under")]


def test_display_name_tie_is_alphabetical():
    train = [_graph(1, [_obj(1, "sofa", "sofa.n.01")]), _graph(2, [_obj(1, "couch", "sofa.n.01")])]
    assert _vocab(train).concepts[0].text == "couch"


def test_junk_name_is_dropped_even_with_a_synset():
    train = [_graph(i, [_obj(1, "he", "helium.n.01"), _obj(2, "man", "man.n.01")]) for i in (1, 2)]
    vocab = _vocab(train)
    assert [c.key for c in vocab.concepts] == ["man.n.01"]
    assert vocab.junk_concept_keys == {"helium.n.01"}


def test_concept_usually_called_by_a_junk_name_is_dropped_with_its_other_names():
    # `side.n.01` is mostly written "side"; "right side" must not survive as its display text.
    train = [_graph(i, [_obj(1, "side", "side.n.01"), _obj(2, "man", "man.n.01")]) for i in (1, 2, 3)]
    train += [_graph(i, [_obj(1, "right side", "side.n.01"), _obj(2, "man", "man.n.01")], [_rel(2, 1)]) for i in (4, 5)]
    vocab = _vocab(train, min_train_images=2)
    assert [c.key for c in vocab.concepts] == ["man.n.01"]
    assert vocab.junk_concept_keys == {"side.n.01"}
    assert vocab.predicates == ()  # `on` only ever touched the junk concept

    _, counts = build_image_graph(train[3], Split.TRAIN, vocab, JUNK)
    assert counts.dropped[GraphDropReason.OBJECT_JUNK_CONCEPT] == 1
    assert counts.dropped[GraphDropReason.OBJECT_JUNK_NAME] == 0


def test_junk_variant_of_a_real_concept_does_not_become_its_display_text():
    # `shirt.n.01` is usually "shirt"; the junk raw name "white" is ignored, the concept stays.
    train = [_graph(i, [_obj(1, "shirt", "garment.n.01")]) for i in (1, 2, 3)]
    train += [_graph(i, [_obj(1, "white", "garment.n.01")]) for i in (4, 5)]
    concept = _vocab(train).concepts[0]
    assert (concept.text, concept.train_images) == ("shirt", 3)


def test_objects_without_synset_become_name_concepts():
    train = [_graph(i, [_obj(1, "street light")]) for i in (1, 2)]
    concept = _vocab(train, min_train_images=2).concepts[0]
    assert (concept.key, concept.text, concept.synset) == ("name:street light", "street light", None)


def test_predicate_counted_only_between_kept_concepts():
    # `near` is in 3 train images, enough on its own, but always touches the rare zebra.
    train = [
        _graph(i, [_obj(1, "man", "man.n.01"), _obj(2, f"zebra {i}", f"zebra.n.0{i}")], [_rel(1, 2, "near")])
        for i in (1, 2, 3)
    ]
    vocab = _vocab(train, min_train_images=3)
    assert [c.key for c in vocab.concepts] == ["man.n.01"]
    assert vocab.predicates == ()
    # Sanity: once the endpoints are frequent enough, the same predicate is kept.
    assert [p.text for p in _vocab(train, min_train_images=1).predicates] == ["near"]


def test_vocabulary_needs_train_images():
    with pytest.raises(DataError, match="zero train"):
        _vocab([])


def test_image_graph_merges_objects_per_concept_and_counts_drops():
    vocab = _vocab([_man_on_couch(1), _man_on_couch(2)], min_train_images=2)
    graph = _graph(
        9,
        [_obj(1, "man", "man.n.01"), _obj(2, "men", "man.n.01"), _obj(3, "sofa", "sofa.n.01"),
         _obj(4, "white"), _obj(5, "zebra", "zebra.n.01")],
        [_rel(1, 3), _rel(2, 3),            # same triplet twice after merging the two men
         _rel(1, 2, "on"),                  # man on man → self loop on the shared node
         _rel(1, 3, "sitting on"),          # predicate not in vocabulary
         _rel(1, 5), _rel(4, 3)],           # endpoint out of vocabulary / junk
    )
    result, counts = build_image_graph(graph, Split.TEST, vocab, JUNK)

    man, couch = vocab.concept_ids()["man.n.01"], vocab.concept_ids()["sofa.n.01"]
    on = vocab.predicate_ids()["on"]
    assert (result.vg_image_id, result.split) == (9, Split.TEST)
    assert dict(zip(result.concept_ids, result.concept_counts)) == {man: 2, couch: 1}
    assert set(result.triplets) == {(man, on, couch), (man, on, man)}
    assert list(result.concept_ids) == sorted(result.concept_ids)
    assert list(result.triplets) == sorted(result.triplets)

    assert (counts.images, counts.objects, counts.objects_kept) == (1, 5, 3)
    assert (counts.relations, counts.triplets_kept, counts.self_loop_triplets) == (6, 2, 1)
    assert counts.dropped == {
        GraphDropReason.OBJECT_JUNK_NAME: 1,
        GraphDropReason.OBJECT_OUT_OF_VOCABULARY: 1,
        GraphDropReason.RELATION_DUPLICATE: 1,
        GraphDropReason.RELATION_PREDICATE_OUT_OF_VOCABULARY: 1,
        GraphDropReason.RELATION_ENDPOINT_DROPPED: 2,
    }
    # Every object and relation is either kept or counted under exactly one reason.
    object_drops = sum(counts.dropped[r] for r in OBJECT_DROP_REASONS)
    assert counts.objects == counts.objects_kept + object_drops
    assert counts.relations == counts.triplets_kept + sum(counts.dropped.values()) - object_drops


def test_image_with_nothing_in_vocabulary_is_kept_empty_and_counted():
    vocab = _vocab([_man_on_couch(1)])
    result, counts = build_image_graph(_graph(5, [_obj(1, "zebra", "zebra.n.01")]), Split.VAL, vocab, JUNK)
    assert (result.concept_ids, result.concept_counts, result.triplets) == ((), (), ())
    assert (counts.images_without_concepts, counts.images_without_triplets) == (1, 1)


def _dataset(eval_graphs=None):
    """Two train images plus the given val/test graphs (default: one val image)."""
    graphs = [_man_on_couch(1), _man_on_couch(2)]
    split_of = {1: Split.TRAIN, 2: Split.TRAIN}
    for image_id, (split, graph) in enumerate(eval_graphs if eval_graphs is not None else [(Split.VAL, _man_on_couch(0))], start=100):
        graphs.append(SceneGraph(image_id, graph.objects, graph.relations))
        split_of[image_id] = split
    return graphs, split_of


def _build(dataset, min_train_images=2):
    return build_graphs(*dataset, min_train_images=min_train_images, junk_names=JUNK)


def test_vocabulary_ignores_validation_and_test_images():
    """Leakage guard: nothing outside train may add concepts, predicates, names or counts."""
    # Eval images call the man "guy", the couch "sofa", and bring a frequent new concept and predicate.
    eval_graph = _graph(
        0,
        [_obj(1, "guy", "man.n.01"), _obj(2, "sofa", "sofa.n.01"), _obj(3, "zebra", "zebra.n.01")],
        [_rel(1, 2, "near"), _rel(1, 3, "near")],
    )
    eval_graphs = [(Split.TEST, eval_graph)] * 6 + [(Split.VAL, eval_graph)] * 6
    base = _build(_dataset())
    with_eval = _build(_dataset(eval_graphs))
    without_eval = _build(_dataset([]))

    assert with_eval.vocabulary == base.vocabulary == without_eval.vocabulary
    assert [(c.key, c.text, c.train_images) for c in with_eval.vocabulary.concepts] == [
        ("man.n.01", "man", 2), ("sofa.n.01", "couch", 2),
    ]
    assert [p.text for p in with_eval.vocabulary.predicates] == ["on"]
    assert with_eval.vocabulary.train_images == 2

    test_graph = next(g for g in with_eval.image_graphs if g.split is Split.TEST)
    assert test_graph.concept_ids == (0, 1)  # guy → man, sofa → couch; zebra dropped
    assert test_graph.triplets == ()  # `near` is not in the train vocabulary
    assert with_eval.counts[Split.TEST].dropped[GraphDropReason.OBJECT_OUT_OF_VOCABULARY] == 6


def test_build_graphs_is_independent_of_input_order_and_tallies_per_split():
    graphs, split_of = _dataset()
    forward = _build((graphs, split_of))
    backward = _build((list(reversed(graphs)), split_of))
    assert forward.vocabulary == backward.vocabulary
    assert forward.image_graphs == backward.image_graphs
    assert [g.vg_image_id for g in forward.image_graphs] == [1, 2, 100]
    assert [(s, forward.counts[s].images, forward.counts[s].objects_kept) for s in Split] == [
        (Split.TRAIN, 2, 4), (Split.VAL, 1, 2), (Split.TEST, 0, 0),
    ]


def test_build_graphs_requires_graphs_and_splits_to_match():
    graphs, split_of = _dataset()
    with pytest.raises(DataError, match="no split"):
        _build((graphs, {k: v for k, v in split_of.items() if k != 100}))
    with pytest.raises(DataError, match="no scene graph"):
        _build((graphs, {**split_of, 999: Split.VAL}))
    with pytest.raises(DataError, match="more than one scene graph"):
        _build((graphs + [graphs[0]], split_of))


def test_graph_statistics_reports_vocabulary_and_splits():
    stats = graph_statistics(_build(_dataset()))
    assert (stats["concepts"], stats["concepts_with_synset"], stats["predicates"]) == (2, 2, 1)
    assert stats["train_images"] == 2
    assert stats["train_images_per_concept"]["median"] == 2
    assert stats["top_concepts_by_train_image_pct"] == [("man", 100.0), ("couch", 100.0)]
    train = stats["by_split"]["train"]
    assert (train["images"], train["objects"], train["objects_kept"]) == (2, 6, 4)
    assert (train["concept_nodes"], train["triplets_kept"], train["relations_kept_pct"]) == (4, 2, 100.0)
    assert train["dropped"]["object_junk_name"] == 2
    assert train["concepts_per_image"]["median"] == 2
    assert stats["by_split"]["test"]["images"] == 0
    assert stats["by_split"]["val"]["concepts_per_image"] == {}  # one image: too few for quantiles


def test_shared_display_text_is_reported():
    # Two different synsets that annotators both call "man".
    train = [_graph(i, [_obj(1, "man", "man.n.01"), _obj(2, "man", "man.n.03")]) for i in (1, 2)]
    stats = graph_statistics(
        build_graphs(train, {1: Split.TRAIN, 2: Split.TRAIN}, min_train_images=1, junk_names=JUNK)
    )
    assert stats["concept_texts_shared"] == 2
    assert stats["shared_text_examples"] == ["man"]
