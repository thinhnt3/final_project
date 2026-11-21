import pytest
import torch

from src.core.exceptions import DataError
from src.embeddings.device import Device
from src.models.kg_train import KgQuerySet
from src.models.relation_paths import (
    PathQuerySet,
    RelationPathConfig,
    RelationPathScorer,
    evaluate_paths,
    fit_paths,
    path_share_loss,
)
from src.models.train import SelectionMetric, TrainingConfig

CHAIN = torch.tensor([[0, 0, 1], [1, 0, 2]])  # 0 → 1 → 2, degrees 1, 2, 1


def _scorer(triples=CHAIN, n_entities=3, n_relations=1, hops=2, dim=4):
    return RelationPathScorer(RelationPathConfig(max_hops=hops), triples, n_entities, n_relations, dim)


def test_untrained_scorer_is_the_degree_normalized_walk_count():
    scores = _scorer()(torch.randn(1, 4), torch.tensor([0]))
    # One step reaches 1 with weight 1/sqrt(1·2); two steps reach 2 with 1/sqrt(2)·1/sqrt(2).
    # The walk 0 → 1 → 0 ends on the anchor, whose column is always 0.
    assert scores[0].tolist() == pytest.approx([0.0, 2**-0.5, 0.5])


def test_hidden_triple_is_removed_for_that_query_only():
    scorer = _scorer()
    scores = scorer(torch.randn(2, 4), torch.tensor([0, 0]), hidden=torch.tensor([[1, 1]]))
    assert scores[0, 2] > 0 and scores[1, 2] == 0  # row 1 lost the edge 1 → 2
    assert scores[1, 1] == scores[0, 1]


def test_scorer_needs_a_graph():
    with pytest.raises(DataError, match="non-empty"):
        _scorer(triples=torch.empty(0, 3, dtype=torch.long))


def test_path_share_loss_leaves_out_skipped_and_unreached():
    scores = torch.tensor([[1.0, 1.0, 2.0], [0.0, 3.0, 1.0]])
    targets = torch.tensor([0, 0])  # the second query's answer is not reached
    no_skip = path_share_loss(scores, targets, torch.zeros(2, 3, dtype=torch.bool))
    assert no_skip.item() == pytest.approx(-torch.log(torch.tensor(0.25)).item())
    skip = torch.tensor([[False, False, True], [False, False, False]])
    assert path_share_loss(scores, targets, skip).item() == pytest.approx(-torch.log(torch.tensor(0.5)).item())


def test_fit_learns_which_relation_a_query_follows():
    # Film i links to person A_i by relations 0 and 1, and to person B_i by relations 2 and 3.
    # Queries of kind A ask for A_i through relation 0, which is hidden while training and
    # absent for validation films, so the model has to learn "kind A follows relation 1".
    n, n_train = 20, 14
    a, b = n, 2 * n
    asked = [[film, 0, a + film] for film in range(n_train)] + [[film, 2, b + film] for film in range(n_train)]
    other = [[film, 1, a + film] for film in range(n)] + [[film, 3, b + film] for film in range(n)]
    triples = torch.tensor(asked + other)
    gallery = torch.arange(n, 3 * n)

    def queries(films, own):
        vectors = torch.eye(2).repeat_interleave(len(films), dim=0)  # kind A rows, then kind B rows
        relevant = [[f] for f in films] + [[n + f] for f in films]
        own_triples = [[f] for f in films] + [[n_train + f] for f in films] if own else [[] for _ in relevant]
        return PathQuerySet(KgQuerySet(vectors, relevant, [[] for _ in relevant], own_triples),
                            torch.tensor(list(films) * 2))

    train, val = queries(range(n_train), own=True), queries(range(n_train, n), own=False)
    config = TrainingConfig(seeds=(1,), device=Device.CPU, batch_size=8, max_epochs=30, patience=30,
                            learning_rate=0.05, weight_decay=0.0, temperature=0.1, grad_clip_norm=1.0,
                            selection_metric=SelectionMetric.RECALL_AT_1)
    model = _scorer(triples, n_entities=3 * n, n_relations=4, hops=1, dim=2)
    assert evaluate_paths(model, val, gallery, (1,))["recall@1"] == 0.0  # A_i and B_i tie
    result = fit_paths(model, gallery, train, val, config, seed=1, ks=(1,))
    assert result.history[0]["epoch"] == 0
    assert result.history[result.best_epoch]["val_recall@1"] == 1.0
