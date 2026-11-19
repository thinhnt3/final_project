import pytest
import torch
from torch_geometric.data import Batch

from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.adamf_mat import MODALITIES, AdamfMatConfig, AdamfMatImageEncoder
from src.models.checkpoint import RunDirectory
from src.models.graph_data import image_graph_to_data
from src.models.graphsage import GraphSageConfig, ImageNodeFeature
from src.models.train import RetrievalSet, SelectionMetric, TrainingConfig, fit, info_nce, seed_everything

DIM = 16
KS = (1, 5)
TEMPERATURE = 0.1

MAN_RIDES_BIKE = {"vg_image_id": 1, "concept_ids": [3, 7], "triplets": [[7, 4, 3]]}
DOG_ONLY = {"vg_image_id": 3, "concept_ids": [9], "triplets": []}
EMPTY = {"vg_image_id": 4, "concept_ids": [], "triplets": []}


def _unit(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(*rows, generator=generator), dim=-1)


def _config(**overrides):
    values = dict(
        dim=12, adaptive_fusion=True, mat=True, mu=1.0,
        noise_dim=4, generator_hidden_dim=8, generator_learning_rate=0.01,
    )
    values.update(overrides)
    return AdamfMatConfig(**values)


def _encoder(seed=0, **overrides):
    seed_everything(seed)
    structure = GraphSageConfig(hidden_dim=20, num_layers=2, directed=True, image_node=ImageNodeFeature.CLIP_IMAGE)
    return AdamfMatImageEncoder(
        _config(**overrides), structure, _unit((20, DIM), 10), _unit((5, DIM), 11), _unit((20, DIM), 12)
    )


def _batch(*graphs_and_rows):
    return Batch.from_data_list([image_graph_to_data(graph, row) for graph, row in graphs_and_rows])


def _dataset(n_images=16, captions=2, seed=1):
    graphs = [
        image_graph_to_data(
            {"vg_image_id": i, "concept_ids": [i, (i + 3) % 20], "triplets": [[i, i % 5, (i + 3) % 20]]}, i
        )
        for i in range(n_images)
    ]
    return RetrievalSet(graphs=graphs, caption_vectors=_unit((n_images, captions, DIM), seed))


def _training(**overrides):
    values = dict(
        seeds=(0,), device=Device.CPU, batch_size=8, max_epochs=3, patience=5, learning_rate=0.01,
        weight_decay=0.0, temperature=TEMPERATURE, grad_clip_norm=1.0, selection_metric=SelectionMetric.RECALL_AT_1,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _fit(tmp_path, name, resume=False, encoder=None, **overrides):
    run = RunDirectory(tmp_path / name)
    run.prepare(resume=resume, overwrite=False)
    model = encoder or _encoder()
    result = fit(model, _dataset(), _dataset(8, seed=2), _training(**overrides), 0, run, KS, resume=resume)
    return result, model, run


def test_output_is_one_unit_vector_per_image_even_without_objects():
    out = _encoder().eval()(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2)))
    assert out.shape == (3, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_modality_weights_sum_to_one_and_differ_between_entities():
    encoder = _encoder().eval()
    weights = encoder.modality_weights(*encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1))))
    assert weights.shape == (2, len(MODALITIES))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert not torch.allclose(weights[0], weights[1], atol=1e-4)


def test_adaptive_fusion_is_softmax_of_the_shared_vector_on_tanh_of_each_modality():
    encoder = _encoder().eval()
    embeddings = encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, 0)))
    scores = torch.cat([torch.tanh(e) @ encoder.attention.weight.T for e in embeddings], dim=-1)
    weights = torch.softmax(scores, dim=-1)
    assert torch.allclose(encoder.modality_weights(*embeddings), weights, atol=1e-6)
    joint = sum(w * e for w, e in zip(weights.unbind(-1), embeddings, strict=True))
    expected = torch.nn.functional.normalize(encoder.project(joint), dim=-1)
    assert torch.allclose(encoder.fuse(*embeddings), expected, atol=1e-6)


def test_without_adaptive_fusion_every_modality_weighs_one_third():
    encoder = _encoder(adaptive_fusion=False).eval()
    weights = encoder.modality_weights(*encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, 0), (EMPTY, 1))))
    assert torch.allclose(weights, torch.full((2, 3), 1 / 3))


def test_image_node_never_carries_the_clip_image_vector_in_the_structure_modality():
    # The structure config asks for clip_image; the image must still enter through its own modality only.
    encoder = _encoder().eval()
    assert encoder.structure.config.image_node is ImageNodeFeature.LEARNED
    first, second = (encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, row))) for row in (0, 5))
    assert torch.equal(first[0], second[0]) and torch.equal(first[2], second[2])
    assert not torch.allclose(first[1], second[1])


def test_discriminator_loss_is_info_nce_plus_mu_times_the_adversarial_term():
    batch, texts = _batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2)), _unit((3, DIM), 3)
    losses = {}
    for mu in (1.0, 3.0):
        encoder = _encoder(mu=mu)
        torch.manual_seed(7)  # same noise for both values of mu
        losses[mu] = encoder.discriminator_loss(batch, texts, TEMPERATURE)
    plain = info_nce(encoder(batch), texts, TEMPERATURE)
    adversarial = losses[1.0] - plain
    assert adversarial > 0
    assert torch.allclose(losses[3.0], plain + 3 * adversarial, atol=1e-5)


def test_generator_loss_reaches_only_the_generators_and_falls_when_they_are_trained():
    encoder = _encoder()
    batch, texts = _batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1), (EMPTY, 2)), _unit((3, DIM), 3)
    generator = list(encoder.generator_parameters())
    generator_ids = {id(p) for p in generator}
    encoder.generator_loss(batch, texts, TEMPERATURE).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in generator)
    structure = [p for name, p in encoder.named_parameters() if name.startswith(("structure", "image_proj"))]
    assert all(p.grad is None for p in structure if id(p) not in generator_ids)

    optimizer = torch.optim.Adam(generator, lr=0.05)
    mean_loss = lambda: torch.stack([encoder.generator_loss(batch, texts, TEMPERATURE) for _ in range(20)]).mean()  # noqa: E731
    before = mean_loss().item()
    for _ in range(100):
        optimizer.zero_grad()
        encoder.generator_loss(batch, texts, TEMPERATURE).backward()
        optimizer.step()
    assert mean_loss().item() < before


def test_training_updates_encoder_and_generator_and_records_both_losses(tmp_path):
    encoder = _encoder()
    before = {name: tensor.clone() for name, tensor in encoder.state_dict().items()}
    result, model, run = _fit(tmp_path, "run", encoder=encoder)
    assert all("train_generator_loss" in row and "train_loss" in row for row in result.history)
    after = model.state_dict()
    for name in ("structure_proj.weight", "attention.weight", "image_generator.layers.0.weight"):
        assert not torch.equal(before[name], after[name]), name
    assert "generator_optimizer" in run.load_last()


def test_resumed_adversarial_run_matches_an_uninterrupted_run(tmp_path):
    full, full_model, _ = _fit(tmp_path, "full", max_epochs=4)
    _fit(tmp_path, "split", max_epochs=2)
    resumed, resumed_model, _ = _fit(tmp_path, "split", resume=True, max_epochs=4)
    strip = lambda history: [{k: v for k, v in row.items() if k != "seconds"} for row in history]  # noqa: E731
    assert strip(resumed.history) == strip(full.history)
    for name, tensor in full_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name


def test_without_mat_there_is_no_generator_and_training_is_plain(tmp_path):
    encoder = _encoder(mat=False)
    assert not encoder.adversarial and list(encoder.generator_parameters()) == []
    assert not any("generator" in name for name in encoder.state_dict())
    result, _, run = _fit(tmp_path, "run", encoder=encoder)
    assert all("train_generator_loss" not in row for row in result.history)
    assert "generator_optimizer" not in run.load_last()


def test_run_names_tell_the_ablations_apart():
    assert _config(dim=256).run_name() == "adamf-mat-d256-mu1"
    assert _config(dim=256, mu=0.5).run_name() == "adamf-mat-d256-mu0.5"
    assert _config(dim=256, mat=False).run_name() == "adamf-d256"
    assert _config(dim=256, adaptive_fusion=False).run_name() == "mean-mat-d256-mu1"


def test_invalid_settings_are_rejected():
    for overrides in ({"dim": 0}, {"noise_dim": 0}, {"mu": 0.0}, {"generator_learning_rate": 0.0}):
        with pytest.raises(ConfigError):
            _config(**overrides)


def test_dropping_a_modality_fuses_only_the_others_and_names_the_run():
    encoder = _encoder(mat=False, modalities=("image", "text")).eval()
    embeddings = encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1)))
    assert len(embeddings) == 2
    assert not hasattr(encoder, "structure")
    out = encoder(_batch((MAN_RIDES_BIKE, 0), (EMPTY, 2)))
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert encoder.config.run_name() == "adamf-d12-no-structure"


def test_mat_needs_every_modality_and_modalities_keep_their_order():
    with pytest.raises(ConfigError):
        _config(mat=True, modalities=("image", "text"))
    with pytest.raises(ConfigError):
        _config(mat=False, modalities=("text", "image"))
    with pytest.raises(ConfigError):
        _config(mat=False, modalities=())


def test_normalized_modalities_are_unit_vectors_before_fusion():
    encoder = _encoder(normalize_modalities=True).eval()
    embeddings = encoder.modality_embeddings(_batch((MAN_RIDES_BIKE, 0), (DOG_ONLY, 1)))
    for embedding in embeddings:
        assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert encoder.config.run_name() == "adamf-mat-d12-mu1-l2"


def test_structure_only_rows_fuse_the_structure_modality_alone_in_training():
    encoder = _encoder(mat=False, structure_only_rate=0.5)
    embeddings = tuple(torch.randn(8, 12) for _ in MODALITIES)
    present = torch.tensor([[True, False, False]] * 4 + [[True, True, True]] * 4)
    weights = encoder.modality_weights(*embeddings, present=present)
    assert torch.allclose(weights[:4], torch.tensor([[1.0, 0.0, 0.0]] * 4))
    assert torch.allclose(weights.sum(dim=1), torch.ones(8))
    torch.manual_seed(0)
    mask = encoder.structure_only_mask(1000, torch.device("cpu"))
    assert mask[:, 0].all() and (mask[:, 1] == mask[:, 2]).all()
    assert 0.4 < (~mask[:, 1]).float().mean().item() < 0.6
    assert encoder.config.run_name() == "adamf-d12-so0.5"


def test_structure_only_rate_is_rejected_with_mat_or_without_structure():
    with pytest.raises(ConfigError):
        _config(structure_only_rate=0.5)  # mat is on
    with pytest.raises(ConfigError):
        _config(mat=False, structure_only_rate=0.5, modalities=("image", "text"))
