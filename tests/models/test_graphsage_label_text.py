import torch
from torch_geometric.data import Batch

from src.models.graph_data import image_graph_to_data
from src.models.graphsage import GraphSageConfig, ImageNodeFeature
from src.models.graphsage_label_text import LabelTextGraphEncoder
from src.models.train import seed_everything

DIM = 16
MAN_RIDES_BIKE = {"vg_image_id": 1, "concept_ids": [3, 7], "triplets": [[7, 4, 3]]}
EMPTY = {"vg_image_id": 4, "concept_ids": [], "triplets": []}


def _unit(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(*rows, generator=generator), dim=-1)


def _encoder(concepts=None):
    seed_everything(0)
    config = GraphSageConfig(hidden_dim=20, num_layers=2, directed=True, image_node=ImageNodeFeature.CLIP_IMAGE)
    concepts = _unit((20, DIM), 10) if concepts is None else concepts
    return LabelTextGraphEncoder(config, 8, concepts, _unit((5, DIM), 11), _unit((20, DIM), 12)).eval()


def _batch(*graphs_and_rows):
    return Batch.from_data_list([image_graph_to_data(graph, row) for graph, row in graphs_and_rows])


def test_output_is_one_unit_vector_per_image_even_without_objects():
    out = _encoder()(_batch((MAN_RIDES_BIKE, 0), (EMPTY, 1)))
    assert out.shape == (2, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_label_text_branch_reads_the_concept_vectors():
    batch = _batch((MAN_RIDES_BIKE, 0))
    encoder = _encoder()
    first = encoder(batch)
    encoder.concept_vectors[3] = -encoder.concept_vectors[3]
    assert not torch.allclose(first, encoder(batch))


def test_hard_negative_partners_share_an_object_pair_but_not_the_triplet():
    from src.models.train_data import hard_negative_partners

    graphs = [
        {"vg_image_id": 1, "concept_ids": [0, 1], "triplets": [[0, 0, 1]]},
        {"vg_image_id": 2, "concept_ids": [0, 1], "triplets": [[0, 1, 1]]},  # same pair, other relation
        {"vg_image_id": 3, "concept_ids": [0, 1], "triplets": [[0, 0, 1]]},  # same triplet: not a negative
        {"vg_image_id": 4, "concept_ids": [0, 2], "triplets": []},
    ]
    partners = hard_negative_partners(graphs, [1, 2, 3, 4], limit=5)
    assert partners[0] == (1,)
    assert partners[1] == (0, 2)
    assert partners[3] == ()
