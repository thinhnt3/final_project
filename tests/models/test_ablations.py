"""Causal controls: topology, deterministic rewiring, and model inputs."""

import copy
from collections import Counter

import pytest
import torch
from torch_geometric.data import Batch

from src.core.exceptions import DataError
from src.models.ablations import Ablation, Corruption, ablate_graph, corrupt_graph, remove_triplet
from src.models.checkpoint import changed_settings
from src.models.graph_data import NodeKind, image_graph_to_data
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature


def graph():
    return {"vg_image_id": 42, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [1, 1, 2], [2, 0, 0], [0, 1, 2]]}


def config(variant):
    return GraphSageConfig(
        8,
        3,
        True,
        ImageNodeFeature.LEARNED,
        variant,
    )


def test_rewire_preserves_degrees_labels_and_source():
    original = graph()
    before = copy.deepcopy(original)
    rewired = ablate_graph(
        original,
        Ablation.RANDOM_EDGES,
        17,
    )
    assert original == before
    assert rewired == ablate_graph(
        original,
        Ablation.RANDOM_EDGES,
        17,
    )
    assert rewired != original
    for column in range(3):
        assert Counter(t[column] for t in rewired["triplets"]) == Counter(t[column] for t in original["triplets"])
    assert any(
        ablate_graph(
            original,
            Ablation.RANDOM_EDGES,
            s,
        )
        != rewired
        for s in range(5)
    )


def test_objects_only_preserves_concepts_and_image_links():
    reduced = ablate_graph(graph(), Ablation.OBJECTS_ONLY)
    data = image_graph_to_data(reduced, 9)
    assert data.num_nodes == 4
    assert data.edge_index.tolist() == [[0, 0, 0], [1, 2, 3]]
    assert data.feature_row.tolist() == [9, 0, 1, 2]


def test_mask_removes_predicate_information_without_changing_topology():
    batch = Batch.from_data_list([image_graph_to_data(graph(), 0)])
    concepts, predicates = torch.randn(3, 4), torch.randn(2, 4)
    model = GraphImageEncoder(
        config(Ablation.DROP_PREDICATES),
        concepts,
        predicates,
    )
    features = model.node_features(batch)
    assert torch.count_nonzero(features[batch.node_kind == NodeKind.RELATION]) == 0
    before = model(batch)
    model.predicate_vectors.fill_(123)
    torch.testing.assert_close(before, model(batch))
    assert torch.isfinite(before).all()
    torch.testing.assert_close(before.norm(dim=-1), torch.ones(1))
    before.sum().backward()
    assert model.project.weight.grad is not None


def test_undirected_is_invariant_to_edge_reversal():
    batch = Batch.from_data_list([image_graph_to_data(graph(), 0)])
    model = GraphImageEncoder(
        config(Ablation.UNDIRECTED),
        torch.randn(3, 4),
        torch.randn(2, 4),
    )
    reversed_batch = batch.clone()
    reversed_batch.edge_index = batch.edge_index.flip(0)
    torch.testing.assert_close(model(batch), model(reversed_batch))


def test_empty_graph_and_all_variants_support_batch_backward():
    for variant in Ablation:
        graphs = [graph(), {"vg_image_id": 8, "concept_ids": [], "triplets": []}]
        batch = Batch.from_data_list([image_graph_to_data(ablate_graph(g, variant), i) for i, g in enumerate(graphs)])
        model = GraphImageEncoder(
            config(variant),
            torch.randn(3, 4),
            torch.randn(2, 4),
        )
        output = model(batch)
        assert output.shape == (2, 4)
        assert torch.isfinite(output).all()
        output.sum().backward()


def test_resume_legacy_full_but_rejects_changed_intervention():
    old = {"graphsage": {"directed": True}}
    current = {"graphsage": {"directed": True, "ablation": "full", "ablation_seed": 0}}
    assert changed_settings(old, current) == []
    current["graphsage"]["ablation"] = "random_edges"
    assert "graphsage" in changed_settings(old, current)
    assert config(Ablation.FULL).run_name() == "directed-learned-l3-h8"
    assert config(Ablation.RANDOM_EDGES).run_name() != config(Ablation.FULL).run_name()


def test_every_variant_trains_checkpoints_and_exports(tmp_path):
    from src.models.checkpoint import RunDirectory
    from src.models.train import RetrievalSet, encode_graphs, fit
    from tests.models.test_train import _config

    for variant in Ablation:
        model = GraphImageEncoder(
            config(variant),
            torch.randn(3, 4),
            torch.randn(2, 4),
        )
        graphs = [image_graph_to_data(ablate_graph({**graph(), "vg_image_id": i}, variant), i) for i in range(4)]
        dataset = RetrievalSet(
            graphs,
            torch.nn.functional.normalize(
                torch.randn(
                    4,
                    2,
                    4,
                ),
                dim=-1,
            ),
        )
        run = RunDirectory(tmp_path / variant.value)
        run.prepare(resume=False, overwrite=False)
        training = _config(max_epochs=1, batch_size=4)
        result = fit(
            model,
            dataset,
            dataset,
            training,
            13,
            run,
            (1,),
        )
        assert len(result.history) == 1
        model.load_state_dict(run.load_best()["model"])
        exported = encode_graphs(
            model,
            graphs,
            training,
        )
        assert exported.shape == (4, 4)
        assert torch.isfinite(exported).all()


def test_run_snapshot_serializes_variant_to_yaml(tmp_path):
    from types import SimpleNamespace

    from scripts.train_graphsage import _settings
    from src.core.config_parsing import read_mapping
    from src.data.yaml_io import write_yaml
    from src.retrieval.config import RetrievalConfig

    for variant in Ablation:
        cfg = RetrievalConfig.from_yaml("configs/retrieval/vg_coco.yaml")
        from dataclasses import replace

        cfg = replace(cfg, graphsage=replace(cfg.graphsage, ablation=variant))
        dataset = SimpleNamespace(graphs=[1, 2])
        settings = _settings(
            cfg,
            "test.yaml",
            "test",
            13,
            {"train": dataset, "val": dataset, "inputs_sha256": {}},
        )
        path = tmp_path / "config.yaml"
        write_yaml(path, settings)
        assert read_mapping(path)["graphsage"]["ablation"] == variant.value


def test_remove_triplet_keeps_objects_and_rejects_absent_triplet():
    original = graph()
    masked = remove_triplet(original, [1, 1, 2])
    assert masked["triplets"] == [[0, 0, 1], [2, 0, 0], [0, 1, 2]]
    assert masked["concept_ids"] == original["concept_ids"]
    assert original["triplets"][1] == [1, 1, 2]
    with pytest.raises(DataError):
        remove_triplet(masked, [1, 1, 2])


def test_clean_corruption_is_identity():
    assert corrupt_graph(graph(), Corruption(), n_concepts=10, seed=0) == graph()


def test_corruption_is_deterministic_and_seeded():
    noise = Corruption(drop_edge=0.5, drop_node=0.3, flip_label=0.5)
    first = corrupt_graph(graph(), noise, n_concepts=10, seed=1)
    assert corrupt_graph(graph(), noise, n_concepts=10, seed=1) == first
    others = [corrupt_graph(graph(), noise, n_concepts=10, seed=s) for s in range(2, 12)]
    assert any(other != first for other in others)


def test_dropping_every_node_drops_every_relation():
    noisy = corrupt_graph(graph(), Corruption(drop_node=1.0), n_concepts=10, seed=0)
    assert noisy["concept_ids"] == [] and noisy["triplets"] == []


def test_flipped_labels_stay_unique_and_relations_follow():
    noisy = corrupt_graph(graph(), Corruption(flip_label=1.0), n_concepts=10, seed=0)
    assert len(set(noisy["concept_ids"])) == 3
    assert not set(noisy["concept_ids"]) & {0, 1, 2}
    assert len(noisy["triplets"]) == 4
    assert all(s in noisy["concept_ids"] and o in noisy["concept_ids"] for s, _, o in noisy["triplets"])
