import pytest
import torch
from torch_geometric.data import Batch

from src.core.exceptions import DataError
from src.models.graph_data import IMAGE_NODE, NodeKind, image_graph_to_data

# Concepts 7 (man), 3 (bicycle), 9 (guitar); predicates 4 (riding), 5 (playing).
GRAPH = {"vg_image_id": 42, "concept_ids": [3, 7, 9], "triplets": [[7, 4, 3], [7, 5, 9]]}


def _edges(data):
    return set(map(tuple, data.edge_index.t().tolist()))


def test_nodes_are_image_then_objects_then_relations():
    data = image_graph_to_data(GRAPH, image_row=11)
    assert data.num_nodes == 1 + 3 + 2
    assert data.node_kind.tolist() == [NodeKind.IMAGE, *[NodeKind.OBJECT] * 3, *[NodeKind.RELATION] * 2]
    # image row, then concept ids in the given order, then predicate ids in triplet order
    assert data.feature_row.tolist() == [11, 3, 7, 9, 4, 5]
    assert data.image_id == 42
    assert data.node_kind[IMAGE_NODE] == NodeKind.IMAGE


def test_edges_are_directed_image_to_object_and_subject_to_relation_to_object():
    data = image_graph_to_data(GRAPH, image_row=0)
    bicycle, man, guitar, riding, playing = 1, 2, 3, 4, 5
    assert _edges(data) == {
        (IMAGE_NODE, bicycle), (IMAGE_NODE, man), (IMAGE_NODE, guitar),
        (man, riding), (riding, bicycle),
        (man, playing), (playing, guitar),
    }
    assert data.edge_index.shape == (2, 7)


def test_swapping_subject_and_object_reverses_the_relation_edges():
    reversed_graph = {**GRAPH, "triplets": [[3, 4, 7]]}  # bicycle riding man
    bicycle, man, riding = 1, 2, 4
    assert {(bicycle, riding), (riding, man)} <= _edges(image_graph_to_data(reversed_graph, 0))


def test_relation_between_two_objects_of_the_same_concept():
    data = image_graph_to_data({"vg_image_id": 1, "concept_ids": [7], "triplets": [[7, 4, 7]]}, 0)
    man, relation = 1, 2
    assert _edges(data) == {(IMAGE_NODE, man), (man, relation), (relation, man)}


def test_image_without_triplets_and_image_without_concepts():
    objects_only = image_graph_to_data({"vg_image_id": 1, "concept_ids": [3, 7], "triplets": []}, 0)
    assert objects_only.num_nodes == 3 and _edges(objects_only) == {(0, 1), (0, 2)}
    empty = image_graph_to_data({"vg_image_id": 2, "concept_ids": [], "triplets": []}, 5)
    assert empty.num_nodes == 1 and empty.edge_index.shape == (2, 0)
    assert empty.node_kind.tolist() == [NodeKind.IMAGE] and empty.feature_row.tolist() == [5]


def test_invalid_graphs_are_rejected():
    with pytest.raises(DataError, match="repeated concept"):
        image_graph_to_data({"vg_image_id": 1, "concept_ids": [3, 3], "triplets": []}, 0)
    with pytest.raises(DataError, match="not in the image"):
        image_graph_to_data({"vg_image_id": 1, "concept_ids": [3], "triplets": [[3, 4, 8]]}, 0)


def test_batching_keeps_sub_graphs_separate_and_image_nodes_findable():
    first = image_graph_to_data(GRAPH, image_row=11)                                             # 6 nodes
    second = image_graph_to_data({"vg_image_id": 43, "concept_ids": [], "triplets": []}, 12)     # 1 node
    third = image_graph_to_data({"vg_image_id": 44, "concept_ids": [9], "triplets": []}, 13)     # 2 nodes
    batch = Batch.from_data_list([first, second, third])

    assert batch.num_nodes == 9 and batch.num_graphs == 3
    image_nodes = batch.ptr[:-1]  # first node of each sub-graph
    assert image_nodes.tolist() == [0, 6, 7]
    assert batch.node_kind[image_nodes].tolist() == [NodeKind.IMAGE] * 3
    # Table rows must survive batching unchanged (PyG shifts attributes named `*index*`).
    assert batch.feature_row[image_nodes].tolist() == [11, 12, 13]
    assert batch.feature_row.tolist() == [11, 3, 7, 9, 4, 5, 12, 13, 9]
    assert batch.image_id.tolist() == [42, 43, 44]
    # No edge joins two sub-graphs.
    source_graph, target_graph = batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]]
    assert torch.equal(source_graph, target_graph)
    # The third sub-graph's only edge was shifted by the 7 nodes before it.
    assert (7, 8) in _edges(batch)
