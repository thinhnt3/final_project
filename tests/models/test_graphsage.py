import pytest
import torch
from torch_geometric.data import Batch

from src.core.exceptions import ConfigError, DataError
from src.models.graph_data import image_graph_to_data
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature

DIM = 16


def _tables(seed=0):
    generator = torch.Generator().manual_seed(seed)
    unit = lambda rows: torch.nn.functional.normalize(torch.randn(rows, DIM, generator=generator), dim=-1)  # noqa: E731
    return unit(10), unit(6), unit(20)  # concepts, predicates, images


def _encoder(directed=True, image_node=ImageNodeFeature.LEARNED, num_layers=3, seed=0):
    concepts, predicates, images = _tables()
    torch.manual_seed(seed)
    config = GraphSageConfig(hidden_dim=12, num_layers=num_layers, directed=directed, image_node=image_node)
    return GraphImageEncoder(config, concepts, predicates, images).eval()


def _batch(*graphs_and_rows):
    return Batch.from_data_list([image_graph_to_data(graph, row) for graph, row in graphs_and_rows])


MAN_RIDES_BIKE = {"vg_image_id": 1, "concept_ids": [3, 7], "triplets": [[7, 4, 3]]}
BIKE_RIDES_MAN = {"vg_image_id": 1, "concept_ids": [3, 7], "triplets": [[3, 4, 7]]}
MAN_NEAR_BIKE = {"vg_image_id": 1, "concept_ids": [3, 7], "triplets": [[7, 5, 3]]}
NO_RELATION = {"vg_image_id": 2, "concept_ids": [3, 7], "triplets": []}
DOG_ONLY = {"vg_image_id": 3, "concept_ids": [9], "triplets": []}
EMPTY = {"vg_image_id": 4, "concept_ids": [], "triplets": []}


def test_each_node_starts_from_the_table_of_its_kind():
    concepts, predicates, images = _tables()
    batch = _batch((MAN_RIDES_BIKE, 5))  # nodes: image, concept 3, concept 7, predicate 4
    with_clip = _encoder(image_node=ImageNodeFeature.CLIP_IMAGE).node_features(batch)
    assert torch.equal(with_clip, torch.stack([images[5], concepts[3], concepts[7], predicates[4]]))
    learned = _encoder(image_node=ImageNodeFeature.LEARNED)
    features = learned.node_features(batch)
    assert torch.equal(features[0], learned.image_token.detach())
    assert torch.equal(features[1:], with_clip[1:])


def test_output_is_one_unit_vector_per_image_with_negative_components():
    out = _encoder()(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2)))
    assert out.shape == (3, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert (out < 0).any() and (out > 0).any()


def test_directed_encoder_tells_which_object_is_the_subject():
    encoder = _encoder(directed=True)
    assert not torch.allclose(encoder(_batch((MAN_RIDES_BIKE, 0))), encoder(_batch((BIKE_RIDES_MAN, 0))), atol=1e-5)


def test_undirected_encoder_ignores_which_object_is_the_subject():
    encoder = _encoder(directed=False)
    assert torch.allclose(encoder(_batch((MAN_RIDES_BIKE, 0))), encoder(_batch((BIKE_RIDES_MAN, 0))), atol=1e-6)


@pytest.mark.parametrize("directed", [True, False])
def test_predicate_and_presence_of_a_relation_change_the_vector(directed):
    encoder = _encoder(directed=directed)
    riding = encoder(_batch((MAN_RIDES_BIKE, 0)))
    assert not torch.allclose(riding, encoder(_batch((MAN_NEAR_BIKE, 0))), atol=1e-5)
    assert not torch.allclose(riding, encoder(_batch((NO_RELATION, 0))), atol=1e-5)


def test_two_layers_know_each_objects_role_and_three_layers_know_who_goes_with_whom():
    # Concepts: 1 man, 2 woman, 3 bike, 4 horse; predicate 4 riding.
    pairing_a = {"vg_image_id": 1, "concept_ids": [1, 2, 3, 4], "triplets": [[1, 4, 3], [2, 4, 4]]}
    pairing_b = {"vg_image_id": 1, "concept_ids": [1, 2, 3, 4], "triplets": [[1, 4, 4], [2, 4, 3]]}
    other_role = {"vg_image_id": 1, "concept_ids": [1, 2, 3, 4], "triplets": [[3, 4, 1], [2, 4, 4]]}
    two_layers, three_layers = _encoder(num_layers=2), _encoder(num_layers=3)
    # After two layers the image knows who rides and what is ridden, but both pairings look alike.
    assert torch.allclose(two_layers(_batch((pairing_a, 0))), two_layers(_batch((pairing_b, 0))), atol=1e-6)
    assert not torch.allclose(two_layers(_batch((pairing_a, 0))), two_layers(_batch((other_role, 0))), atol=1e-5)
    # The third layer links the two ends of each relation.
    assert not torch.allclose(three_layers(_batch((pairing_a, 0))), three_layers(_batch((pairing_b, 0))), atol=1e-5)


def test_sub_graphs_in_one_batch_do_not_influence_each_other():
    encoder = _encoder()
    together = encoder(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2), (NO_RELATION, 3)))
    alone = torch.cat([encoder(_batch((g, r))) for g, r in
                       ((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2), (NO_RELATION, 3))])
    assert torch.allclose(together, alone, atol=1e-6)


def test_learned_image_node_ignores_the_image_row_and_clip_image_node_uses_it():
    learned = _encoder(image_node=ImageNodeFeature.LEARNED)
    assert torch.allclose(learned(_batch((MAN_RIDES_BIKE, 0))), learned(_batch((MAN_RIDES_BIKE, 5))), atol=1e-6)
    with_clip = _encoder(image_node=ImageNodeFeature.CLIP_IMAGE)
    assert not torch.allclose(with_clip(_batch((MAN_RIDES_BIKE, 0))), with_clip(_batch((MAN_RIDES_BIKE, 5))), atol=1e-5)


def test_only_layers_projection_and_image_token_are_trainable_and_tables_stay_out_of_checkpoints():
    encoder = _encoder()
    trainable = {name.split(".")[0] for name, p in encoder.named_parameters() if p.requires_grad}
    assert trainable == {"layers", "project", "image_token"}
    assert not any("vectors" in name for name in encoder.state_dict())
    assert "image_token" not in dict(_encoder(image_node=ImageNodeFeature.CLIP_IMAGE).named_parameters())


def test_gradient_reaches_every_layer_from_the_image_vector():
    encoder = _encoder().train()
    encoder(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1))).sum().backward()

    def has_gradient(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    # Earlier layers feed object and relation nodes, so both directions are used there.
    for layer in encoder.layers[:-1]:
        assert has_gradient(layer.forward_conv) and has_gradient(layer.backward_conv)
    # In the last layer only the image node is read, and it has outgoing edges only.
    assert has_gradient(encoder.layers[-1].backward_conv)
    assert has_gradient(encoder.project) and encoder.image_token.grad.abs().sum() > 0


def test_invalid_configuration_is_rejected():
    with pytest.raises(ConfigError, match="must be >= 1"):
        GraphSageConfig(hidden_dim=8, num_layers=0, directed=True, image_node=ImageNodeFeature.LEARNED)
    concepts, predicates, _ = _tables()
    config = GraphSageConfig(hidden_dim=8, num_layers=2, directed=True, image_node=ImageNodeFeature.CLIP_IMAGE)
    with pytest.raises(DataError, match="image vectors"):
        GraphImageEncoder(config, concepts, predicates, None)
    with pytest.raises(DataError, match="differ in size"):
        GraphImageEncoder(config, concepts, predicates[:, :8], None)
