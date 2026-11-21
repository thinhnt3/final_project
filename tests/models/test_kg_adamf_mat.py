"""AdaMF-MAT on a knowledge graph: fusion weights, RotatE scores, losses and the training loop."""
from __future__ import annotations

import pytest
import torch

from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.kg_adamf_mat import KgAdamfMat, KgAdamfMatConfig, sigmoid_loss
from src.models.kg_adamf_train import KgAdamfTrainingConfig, fit_kg_adamf

N_ENTITIES, N_RELATIONS, CLIP_DIM = 12, 3, 8


def _config(**changes: object) -> KgAdamfMatConfig:
    values: dict[str, object] = {
        "dim": 6, "margin": 4.0, "epsilon": 2.0, "negatives": 5, "adversarial_temperature": 1.0,
        "adaptive_fusion": True, "mat": True, "mu": 1.0, "noise_dim": 4, "generator_hidden_dim": 8,
        "generator_learning_rate": 1e-3,
    }
    return KgAdamfMatConfig(**{**values, **changes})  # type: ignore[arg-type]


def _model(**changes: object) -> KgAdamfMat:
    torch.manual_seed(0)
    return KgAdamfMat(_config(**changes), N_ENTITIES, N_RELATIONS, torch.randn(N_ENTITIES, CLIP_DIM),
                      torch.randn(N_ENTITIES, CLIP_DIM))


def test_fusion_weights_sum_to_one_and_mean_fusion_is_uniform() -> None:
    ids = torch.arange(N_ENTITIES)
    adaptive = _model()
    weights = adaptive.modality_weights(*adaptive.modality_embeddings(ids))
    assert weights.shape == (N_ENTITIES, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(N_ENTITIES))
    mean = _model(adaptive_fusion=False, mat=False)
    assert torch.allclose(mean.modality_weights(*mean.modality_embeddings(ids)), torch.full((N_ENTITIES, 3), 1 / 3))


def test_structure_only_joint_is_the_entity_embedding() -> None:
    model = _model(modalities=("structure",), mat=False)
    ids = torch.tensor([0, 5])
    assert torch.allclose(model.joint(ids), model.entities(ids))
    assert model.config.run_name() == "rotate"


def test_score_is_margin_when_the_rotated_head_equals_the_tail() -> None:
    model = _model(mat=False)
    with torch.no_grad():
        model.relations.weight.zero_()  # zero phase: the rotation is the identity
    same = torch.randn(2, 1, 12)
    scores = model.score(same, torch.tensor([0, 1]), same)
    assert torch.allclose(scores, torch.full((2, 1), 4.0), atol=1e-4)
    assert (model.score(same, torch.tensor([0, 1]), same + 1.0) < scores).all()


def test_candidate_scores_follow_the_anchor_side() -> None:
    model = _model(mat=False)
    anchors, relations, everyone = torch.tensor([1, 2]), torch.tensor([0, 2]), torch.arange(N_ENTITIES)
    scores = model.candidate_scores(anchors, relations, torch.tensor([True, False]), everyone)
    joint = model.joint(everyone)
    as_tail = model.score(joint[1].view(1, 1, -1), relations[:1], joint[7].view(1, 1, -1))
    as_head = model.score(joint[7].view(1, 1, -1), relations[1:], joint[2].view(1, 1, -1))
    assert scores.shape == (2, N_ENTITIES)
    assert torch.allclose(scores[0, 7], as_tail.squeeze(), atol=1e-5)
    assert torch.allclose(scores[1, 7], as_head.squeeze(), atol=1e-5)


def test_sigmoid_loss_prefers_positives_above_negatives() -> None:
    good = sigmoid_loss(torch.tensor([[3.0]]), torch.tensor([[-3.0, -2.0]]), 1.0)
    bad = sigmoid_loss(torch.tensor([[-3.0]]), torch.tensor([[3.0, 2.0]]), 1.0)
    assert good < bad


def test_generator_loss_reaches_only_the_generators() -> None:
    model = _model()
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]])
    model.generator_loss(triples).backward()
    generator_ids = {id(p) for p in model.generator_parameters()}
    assert all(p.grad is not None for p in model.generator_parameters())
    assert generator_ids.isdisjoint(id(p) for p in model.encoder_parameters())
    without = _model(mat=False)
    assert list(without.generator_parameters()) == []


def test_config_rejects_mat_without_all_modalities_and_missing_structure() -> None:
    with pytest.raises(ConfigError):
        _config(modalities=("structure", "image"))
    with pytest.raises(ConfigError):
        _config(modalities=("image", "text"), mat=False)
    assert _config(modalities=("structure", "text"), mat=False).run_name() == "adamf-no-image"
    assert _config().run_name() == "adamf-mat-mu1"


def test_fit_learns_a_small_graph() -> None:
    # A chain: relation 0 links entity i to i + 1. The model should rank the true tail first.
    train = [[i, 0, i + 1] for i in range(N_ENTITIES - 1)]
    model = _model(mat=True)
    config = KgAdamfTrainingConfig(device=Device.CPU, batch_size=4, learning_rate=0.05, max_epochs=60, eval_every=20,
                                   patience=3, eval_batch_size=8)
    result = fit_kg_adamf(model, train, train, train, config, seed=1)
    assert result.history[-1]["val_mrr"] > result.history[0]["val_mrr"] or result.history[0]["val_mrr"] > 0.9
    assert max(entry["val_mrr"] for entry in result.history) > 0.6
    assert "generator_loss" in result.history[0]
