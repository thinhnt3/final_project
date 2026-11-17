from collections import Counter
from copy import deepcopy

import pytest
from torch_geometric.data import Batch

from src.core.exceptions import ConfigError, DataError
from src.models.ablations import objects_only, remove_relations, rewire_relations
from src.models.graph_data import NodeKind, image_graph_to_data

GRAPH = {"vg_image_id": 7, "concept_ids": [0, 1, 2, 3], "triplets": [[0, 4, 1], [2, 5, 3]]}


def test_objects_only_removes_relation_nodes_but_keeps_objects_and_image_edges():
    original = deepcopy(GRAPH)
    changed = objects_only(GRAPH)
    data = image_graph_to_data(changed, 8)
    assert original == GRAPH
    assert changed["concept_ids"] == GRAPH["concept_ids"]
    assert data.node_kind.tolist() == [NodeKind.IMAGE, *[NodeKind.OBJECT] * 4]
    assert data.feature_row.tolist() == [8, 0, 1, 2, 3]
    assert data.edge_index.tolist() == [[0, 0, 0, 0], [1, 2, 3, 4]]


def test_removal_uses_relation_position_including_duplicate_triplets():
    graph = {**GRAPH, "triplets": [[0, 4, 1], [0, 4, 1], [2, 5, 3]]}
    assert remove_relations(graph, [1])["triplets"] == [[0, 4, 1], [2, 5, 3]]
    with pytest.raises(DataError, match="index"):
        remove_relations(graph, [3])
    with pytest.raises(DataError, match="index"):
        remove_relations(graph, [True])


def test_target_swap_preserves_degrees_predicates_and_inputs():
    original = deepcopy(GRAPH)
    result = rewire_relations(GRAPH, seed=13, swaps=1, max_attempts=10)
    assert original == GRAPH
    assert result.successful_swaps == 1 and result.changed_relations == 2
    assert result.graph["triplets"] == [[0, 4, 3], [2, 5, 1]]
    for position in (0, 1, 2):
        assert Counter(t[position] for t in GRAPH["triplets"]) == Counter(t[position] for t in result.graph["triplets"])
    assert result == rewire_relations(GRAPH, seed=13, swaps=1, max_attempts=10)
    data = image_graph_to_data(result.graph, 8)
    assert data.num_nodes == 7 and data.edge_index.shape[1] == 8


def test_no_valid_swap_reports_failure_without_changing_graph():
    graph = {**GRAPH, "triplets": [[0, 4, 1], [0, 5, 3]]}
    result = rewire_relations(graph, seed=13, swaps=2, max_attempts=5)
    assert result.graph == graph
    assert result.successful_swaps == result.changed_relations == 0
    assert result.attempts == 5 and result.requested_swaps == 2


def test_rewiring_does_not_introduce_self_loops_or_duplicates():
    graph = {**GRAPH, "triplets": [[0, 4, 1], [1, 4, 0], [0, 4, 3], [2, 4, 1]]}
    result = rewire_relations(graph, seed=37, swaps=20, max_attempts=100)
    triples = list(map(tuple, result.graph["triplets"]))
    assert len(triples) == len(set(triples))
    assert all(s != o for s, _, o in triples)


def test_empty_graph_and_zero_budget_are_valid():
    graph = {"vg_image_id": 8, "concept_ids": [], "triplets": []}
    result = rewire_relations(graph, seed=13, swaps=1, max_attempts=5)
    assert result.graph == graph and result.attempts == 0
    assert objects_only(graph) == graph
    assert rewire_relations(GRAPH, seed=13, swaps=0, max_attempts=0).graph == GRAPH
    with pytest.raises(ConfigError):
        rewire_relations(GRAPH, seed=13, swaps=-1, max_attempts=0)


def test_variants_batch_without_cross_graph_edges():
    variants = [objects_only(GRAPH), rewire_relations(GRAPH, seed=13, swaps=1, max_attempts=10).graph]
    batch = Batch.from_data_list([image_graph_to_data(g, i) for i, g in enumerate(variants)])
    source, target = batch.edge_index
    assert (batch.batch[source] == batch.batch[target]).all()
