import pytest
import torch

from src.core.exceptions import ConfigError, DataError
from src.embeddings.device import Device
from src.models.kg_graphsage import KgEntityEncoder, KgSageConfig, RelationSAGELayer, mean_at
from src.models.kg_train import KgQuerySet, _batch_inputs, evaluate, fit, gallery_loss
from src.models.train import SelectionMetric, TrainingConfig


def test_mean_at_averages_per_target():
    messages = torch.tensor([[2.0], [4.0], [10.0]])
    assert mean_at(messages, torch.tensor([0, 0, 2]), 3).squeeze(1).tolist() == [3.0, 0.0, 10.0]


def test_layer_reads_direction_and_relation():
    torch.manual_seed(0)
    layer = RelationSAGELayer(4, 4, n_relations=2)
    x = torch.randn(3, 4)
    base = layer(x, torch.tensor([[0, 0, 1]]))
    assert not torch.allclose(base, layer(x, torch.tensor([[1, 0, 0]])))  # reversed edge
    assert not torch.allclose(base, layer(x, torch.tensor([[0, 1, 1]])))  # other relation
    assert torch.allclose(base[2], layer.own(x)[2])  # node 2 has no edge: only its own state


def test_encoder_reaches_one_hop_per_layer():
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    chain = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    changed = features.clone()
    changed[3] = -features[3]
    for layers, reaches in ((1, False), (3, True)):
        encoder = KgEntityEncoder(KgSageConfig(hidden_dim=8, num_layers=layers), features, n_relations=1)
        before = encoder(chain)[0]
        encoder.entity_features = changed
        assert (not torch.allclose(before, encoder(chain)[0])) is reaches
    assert torch.allclose(encoder(chain).norm(dim=1), torch.ones(4), atol=1e-5)


def test_gallery_loss_ignores_skipped_columns():
    queries = torch.eye(3)[:1]
    gallery = torch.eye(3)
    targets = torch.tensor([1])
    no_skip = gallery_loss(queries, gallery, targets, torch.zeros(1, 3, dtype=torch.bool), 0.1)
    skip = torch.tensor([[True, False, False]])  # column 0 scores highest but is another right answer
    assert gallery_loss(queries, gallery, targets, skip, 0.1) < no_skip


def test_batch_inputs_mask_own_triples_and_other_answers():
    data = KgQuerySet(torch.eye(2), relevant=[[0, 2], [1]], known=[[3], []], own_triples=[[5, 6], []])
    targets, skip, keep = _batch_inputs(data, torch.tensor([0, 1]), n_gallery=4, n_triples=8,
                                        generator=torch.Generator().manual_seed(0))
    assert targets[0].item() in (0, 2) and targets[1].item() == 1
    assert skip[0].tolist() == [c != targets[0].item() and c in (0, 2, 3) for c in range(4)]
    assert not skip[1].any()
    assert keep.tolist() == [True] * 5 + [False, False, True]


def test_query_set_needs_answers():
    with pytest.raises(DataError, match="answer"):
        KgQuerySet(torch.eye(2), relevant=[[0], []], known=[[], []], own_triples=[[], []])


def test_fit_learns_a_one_hop_pattern():
    # 40 "films", each linked to its own "director" (ids 40-79) by a writer edge. Train films
    # also have the asked director edge in the graph, which is masked while they are trained on.
    # A query is a noisy copy of the film's feature; the director's own feature says nothing
    # about the film, so only the graph can connect them.
    torch.manual_seed(0)
    n, n_train = 40, 30
    features = torch.nn.functional.normalize(torch.randn(2 * n, 16), dim=1)
    director = torch.tensor([[film, 0, n + film] for film in range(n_train)])  # rows 0..29
    writer = torch.tensor([[film, 1, n + film] for film in range(n)])
    triples = torch.cat([director, writer])
    gallery = torch.arange(n, 2 * n)

    def queries(films, own):
        vectors = torch.nn.functional.normalize(features[films] + 0.05 * torch.randn(len(films), 16), dim=1)
        return KgQuerySet(vectors, relevant=[[f] for f in films], known=[[] for _ in films],
                          own_triples=[[f] if own else [] for f in films])

    train, val = queries(list(range(n_train)), own=True), queries(list(range(n_train, n)), own=False)
    config = TrainingConfig(seeds=(1,), device=Device.CPU, batch_size=10, max_epochs=40, patience=40,
                            learning_rate=0.01, weight_decay=0.0, temperature=0.1, grad_clip_norm=1.0,
                            selection_metric=SelectionMetric.RECALL_AT_1)
    model = KgEntityEncoder(KgSageConfig(hidden_dim=16, num_layers=1), features, n_relations=2)
    untrained = evaluate(model, triples, gallery, val, (1,))["mrr"]
    result = fit(model, triples, gallery, train, val, config, seed=1, ks=(1,))
    assert result.history[result.best_epoch - 1]["val_mrr"] > max(untrained, 0.6)
    assert result.entity_vectors.shape == (2 * n, 16)


def test_structure_only_rows_apply_in_training_only():
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [3, 0, 4]])
    encoder = KgEntityEncoder(KgSageConfig(hidden_dim=8, num_layers=1, structure_only_rate=0.5), features, 1)
    encoder.eval()
    full = encoder(triples)
    assert torch.equal(full, encoder(triples))
    encoder.train()
    torch.manual_seed(1)
    partial = encoder(triples)
    changed = ~torch.isclose(full, partial).all(dim=1)
    assert changed.any() and not changed.all()
    with pytest.raises(ConfigError, match="structure_only_rate"):
        KgSageConfig(hidden_dim=8, num_layers=1, structure_only_rate=1.0)
