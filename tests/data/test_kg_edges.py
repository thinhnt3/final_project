import pytest

from src.core.exceptions import DataError
from src.data.kg_edges import ISA_PREDICATE, add_isa_edges, build_indirect_queries, extend_vocabulary

CHAINS = {"horse.n.01": ["equine.n.01", "animal.n.01"], "zebra.n.01": ["equine.n.01", "animal.n.01"]}


def _chain(synset, depth):
    return CHAINS.get(synset, [])[:depth]


def _vocab():
    concepts = [
        {"concept_id": 0, "key": "man.n.01", "text": "man", "synset": "man.n.01"},
        {"concept_id": 1, "key": "horse.n.01", "text": "horse", "synset": "horse.n.01"},
        {"concept_id": 2, "key": "zebra.n.01", "text": "zebra", "synset": "zebra.n.01"},
        {"concept_id": 3, "key": "grass", "text": "grass", "synset": None},
    ]
    predicates = [{"predicate_id": 0, "text": "riding"}, {"predicate_id": 1, "text": "near"}]
    return extend_vocabulary(concepts, predicates, _chain, 2, lambda s: s.split(".")[0])


def test_hypernyms_become_shared_concepts_and_isa_a_predicate():
    concepts, predicates, chains = _vocab()
    assert [c["text"] for c in concepts[4:]] == ["equine", "animal"]
    assert chains == {0: [], 1: [4, 5], 2: [4, 5]}
    assert predicates[-1] == {"predicate_id": 2, "text": ISA_PREDICATE, "train_images": 0}
    with pytest.raises(DataError):
        extend_vocabulary([], predicates, _chain, 2, str)  # "is a" already there


def test_isa_edges_add_each_chain_once():
    _, _, chains = _vocab()
    graph = {"vg_image_id": 1, "concept_ids": [0, 1, 2], "concept_counts": [1, 1, 1], "triplets": [[0, 0, 1]]}
    out = add_isa_edges(graph, chains, isa_id=2)
    assert out["concept_ids"] == [0, 1, 2, 4, 5] and out["concept_counts"] == [1, 1, 1, 0, 0]
    assert out["triplets"] == [[0, 0, 1], [1, 2, 4], [2, 2, 4], [4, 2, 5]]
    assert graph["triplets"] == [[0, 0, 1]]  # input untouched


def test_indirect_query_is_unique_under_the_hypernym_and_has_confusers():
    concepts, predicates, chains = _vocab()
    text = {c["concept_id"]: c["text"] for c in concepts}
    ptext = {p["predicate_id"]: p["text"] for p in predicates}
    graphs = [
        {"vg_image_id": 1, "split": "test", "concept_ids": [0, 1], "triplets": [[1, 1, 0]]},  # horse near man
        {"vg_image_id": 2, "split": "test", "concept_ids": [0, 2], "triplets": []},           # zebra and man
    ]
    queries = build_indirect_queries(graphs, {1: [4], 2: [4]}, text, ptext, seed=0)
    by_kind = {q["kind"]: q for q in queries}
    assert by_kind["indirect"]["text"] == "equine near man"
    assert by_kind["direct"]["text"] == "horse near man"
    assert by_kind["indirect"]["confuser_ids"] == [2]
