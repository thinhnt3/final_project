import math

import numpy as np
import pytest
import torch

import src.models.train as train_module
from src.core.exceptions import ConfigError, DataError, TrainingError
from src.data.schemas import Split
from src.embeddings.device import Device
from src.embeddings.store import EmbeddingTable, l2_normalize
from src.models.checkpoint import (
    BEST_FILE,
    HISTORY_FILE,
    LAST_FILE,
    RESULT_FILE,
    SETTINGS_FILE,
    RunDirectory,
    changed_settings,
)
from src.models.graph_data import image_graph_to_data
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature
from src.models.train import (
    RetrievalSet,
    SelectionMetric,
    StopReason,
    TrainingConfig,
    epoch_batches,
    evaluate,
    fit,
    info_nce,
    seed_everything,
    selection_key,
)
from src.models.train_data import build_retrieval_set

DIM = 16
KS = (1, 5)


def _unit(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(*rows, generator=generator), dim=-1)


def _config(**overrides):
    values = dict(
        seeds=(0,),
        device=Device.CPU,
        batch_size=8,
        max_epochs=3,
        patience=5,
        learning_rate=0.01,
        weight_decay=0.0,
        temperature=0.1,
        grad_clip_norm=1.0,
        selection_metric=SelectionMetric.RECALL_AT_1,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _dataset(n_images=16, captions=2, seed=1):
    """Images with pairwise different graphs, so a model can tell them apart."""
    graphs = [
        image_graph_to_data(
            {"vg_image_id": i, "concept_ids": [i, (i + 3) % 20], "triplets": [[i, i % 5, (i + 3) % 20]]}, i
        )
        for i in range(n_images)
    ]
    return RetrievalSet(graphs=graphs, caption_vectors=_unit((n_images, captions, DIM), seed))


def _model(seed=0):
    seed_everything(seed)
    config = GraphSageConfig(hidden_dim=32, num_layers=2, directed=True, image_node=ImageNodeFeature.LEARNED)
    return GraphImageEncoder(config, _unit((20, DIM), 10), _unit((5, DIM), 11))


def _rows(history):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in history]


def _fit(tmp_path, name="run", seed=0, resume=False, **overrides):
    run = RunDirectory(tmp_path / name)
    run.prepare(resume=resume, overwrite=False)
    model = _model(seed)
    result = fit(model, _dataset(), _dataset(n_images=8, seed=2), _config(**overrides), seed, run, KS, resume=resume)
    return result, model, run


def test_info_nce_is_low_for_matching_pairs_and_high_for_shuffled_ones():
    vectors = torch.eye(4)
    assert info_nce(vectors, vectors, temperature=0.05).item() < 1e-6
    assert info_nce(vectors, vectors.roll(1, dims=0), temperature=0.05).item() > 10
    # Uniform similarity: both directions give ln(batch size).
    same = torch.nn.functional.normalize(torch.ones(4, 4), dim=-1)
    assert info_nce(same, same, temperature=1.0).item() == pytest.approx(math.log(4))


def test_info_nce_averages_both_directions():
    graphs, texts = _unit((3, DIM), 1), _unit((3, DIM), 2)
    logits = graphs @ texts.T / 0.2
    targets = torch.arange(3)
    expected = (
        torch.nn.functional.cross_entropy(logits, targets) + torch.nn.functional.cross_entropy(logits.T, targets)
    ) / 2
    assert info_nce(graphs, texts, 0.2).item() == pytest.approx(expected.item())
    assert info_nce(graphs, texts, 0.2).item() == pytest.approx(info_nce(texts, graphs, 0.2).item())
    with pytest.raises(DataError, match="shapes differ"):
        info_nce(graphs, texts[:2], 0.2)


def test_epoch_batches_use_each_image_once_and_depend_only_on_seed_and_epoch():
    batches = epoch_batches(n_images=10, captions_per_image=5, batch_size=4, seed=3, epoch=2)
    images = torch.cat([b[0] for b in batches])
    assert sorted(images.tolist()) == list(range(10))
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    captions = torch.cat([b[1] for b in batches])
    assert captions.min() >= 0 and captions.max() < 5 and len(captions) == 10
    again = epoch_batches(10, 5, 4, seed=3, epoch=2)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(batches, again, strict=True))
    assert not torch.equal(images, torch.cat([b[0] for b in epoch_batches(10, 5, 4, seed=3, epoch=3)]))
    assert not torch.equal(images, torch.cat([b[0] for b in epoch_batches(10, 5, 4, seed=4, epoch=2)]))


def test_model_can_memorize_a_small_set(tmp_path):
    run = RunDirectory(tmp_path / "overfit")
    run.prepare(resume=False, overwrite=False)
    data = _dataset(n_images=16, captions=1)
    model = _model()
    config = _config(max_epochs=150, patience=150, batch_size=16)
    result = fit(model, data, data, config, 0, run, KS)
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"] / 5
    assert evaluate(model, data, config, KS)["recall@1"] == 1.0


def test_same_seed_gives_the_same_history_and_another_seed_does_not(tmp_path):
    first, _, _ = _fit(tmp_path, "a")
    second, _, _ = _fit(tmp_path, "b")
    other, _, _ = _fit(tmp_path, "c", seed=1)
    assert _rows(first.history) == _rows(second.history)
    assert _rows(first.history) != _rows(other.history)


def test_resumed_run_matches_an_uninterrupted_run(tmp_path):
    full, full_model, _ = _fit(tmp_path, "full", max_epochs=4)
    _fit(tmp_path, "split", max_epochs=2)
    resumed, resumed_model, run = _fit(tmp_path, "split", resume=True, max_epochs=4)
    assert _rows(resumed.history) == _rows(full.history)
    assert [row["epoch"] for row in resumed.history] == [1, 2, 3, 4]
    for name, tensor in full_model.state_dict().items():
        assert torch.allclose(tensor, resumed_model.state_dict()[name], atol=1e-7), name
    assert _rows(run.load_history()) == _rows(full.history)


def test_resume_with_another_seed_is_rejected(tmp_path):
    _fit(tmp_path, "run", max_epochs=1)
    run = RunDirectory(tmp_path / "run")
    with pytest.raises(TrainingError, match="seed"):
        fit(_model(1), _dataset(), _dataset(8, seed=2), _config(), 1, run, KS, resume=True)


def test_every_epoch_writes_history_and_checkpoints_and_reports_to_the_callback(tmp_path):
    seen = []
    run = RunDirectory(tmp_path / "run")
    run.prepare(resume=False, overwrite=False)
    result = fit(_model(), _dataset(), _dataset(8, seed=2), _config(max_epochs=3), 0, run, KS, on_epoch=seen.append)
    assert [row["epoch"] for row in seen] == [1, 2, 3] == [row["epoch"] for row in run.load_history()]
    assert {LAST_FILE, BEST_FILE, HISTORY_FILE} <= {p.name for p in run.path.iterdir()}
    assert not list(run.path.glob("*.tmp"))
    row = result.history[0]
    assert {
        "train_loss",
        "val_loss",
        "val_recall@1",
        "val_recall@5",
        "val_mrr",
        "learning_rate",
        "seconds",
        "is_best",
    } <= set(row)
    assert run.load_best()["epoch"] == result.best_epoch
    assert sum(r["is_best"] for r in result.history) >= 1 and result.history[result.best_epoch - 1]["is_best"]
    assert result.stop_reason is StopReason.MAX_EPOCHS


def test_training_stops_after_patience_epochs_without_improvement(tmp_path):
    # A vanishing learning rate keeps validation metrics constant, so epoch 1 stays the best.
    result, _, _ = _fit(tmp_path, max_epochs=20, patience=2, learning_rate=1e-12)
    assert result.best_epoch == 1
    assert [row["epoch"] for row in result.history] == [1, 2, 3]
    assert result.stop_reason is StopReason.NO_IMPROVEMENT


def test_non_finite_loss_stops_training_and_keeps_the_earlier_checkpoints(tmp_path, monkeypatch):
    real_loss = train_module._pair_loss
    calls = {"n": 0}

    def loss_that_diverges(*args, **kwargs):
        calls["n"] += 1
        loss = real_loss(*args, **kwargs)
        return loss * float("nan") if calls["n"] > 2 else loss  # 2 batches per epoch: epoch 2 diverges

    monkeypatch.setattr(train_module, "_pair_loss", loss_that_diverges)
    run = RunDirectory(tmp_path / "run")
    run.prepare(resume=False, overwrite=False)
    model = _model()
    with pytest.raises(TrainingError, match="loss is nan at epoch 2"):
        fit(model, _dataset(), _dataset(8, seed=2), _config(max_epochs=5), 0, run, KS)
    assert [row["epoch"] for row in run.load_history()] == [1]
    assert run.load_best()["epoch"] == 1 and len(run.load_last()["history"]) == 1
    # The weights were not updated by the diverged batch.
    for name, tensor in run.load_last()["model"].items():
        assert torch.equal(tensor, model.state_dict()[name]), name


def test_selection_key_uses_the_chosen_metric_first():
    metrics = {"recall@1": 0.2, "mrr": 0.5}
    assert selection_key(metrics, SelectionMetric.RECALL_AT_1) == (0.2, 0.5)
    assert selection_key(metrics, SelectionMetric.MRR) == (0.5, 0.2)


def test_evaluate_scores_every_caption_against_every_image():
    data = _dataset(n_images=8, captions=2)
    metrics = evaluate(_model(), data, _config(), KS)
    assert set(metrics) == {"loss", "recall@1", "recall@5", "mrr"}
    assert 0.0 <= metrics["recall@1"] <= metrics["recall@5"] <= 1.0 and math.isfinite(metrics["loss"])


def test_run_directory_refuses_to_reuse_a_run_unless_asked(tmp_path):
    run = RunDirectory(tmp_path / "run")
    with pytest.raises(TrainingError, match="cannot resume"):
        run.prepare(resume=True, overwrite=False)
    run.prepare(resume=False, overwrite=False)
    run.save_last({"x": 1})
    with pytest.raises(TrainingError, match="already holds a run"):
        run.prepare(resume=False, overwrite=False)
    run.prepare(resume=True, overwrite=False)
    run.prepare(resume=False, overwrite=True)
    with pytest.raises(TrainingError, match="checkpoint not found"):
        run.load_best()


def test_invalid_training_settings_and_sets_are_rejected():
    with pytest.raises(ConfigError, match="seeds"):
        _config(seeds=())
    with pytest.raises(ConfigError, match="must name a device"):
        _config(device=Device.AUTO)
    with pytest.raises(DataError, match="caption vectors have shape"):
        RetrievalSet(graphs=_dataset().graphs, caption_vectors=_unit((3, 2, DIM), 0))
    with pytest.raises(DataError, match="at least two"):
        RetrievalSet(graphs=_dataset().graphs[:1], caption_vectors=_unit((1, 2, DIM), 0))


def _tables():
    images = EmbeddingTable(ids=(30, 10, 20), vectors=l2_normalize(np.eye(3, 4, dtype=np.float32)))
    query_ids = ("q-b", "q-a", "q-c", "q-d", "q-e", "q-f")
    queries = EmbeddingTable(ids=query_ids, vectors=l2_normalize(np.arange(1, 25, dtype=np.float32).reshape(6, 4)))
    return images, queries


IMAGES = [
    {"vg_image_id": 20, "split": "train"},
    {"vg_image_id": 10, "split": "train"},
    {"vg_image_id": 30, "split": "val"},
]
QUERIES = [
    {"query_id": "q-b", "relevant_ids": [10], "split": "train"},
    {"query_id": "q-a", "relevant_ids": [10], "split": "train"},
    {"query_id": "q-c", "relevant_ids": [20], "split": "train"},
    {"query_id": "q-d", "relevant_ids": [20], "split": "train"},
    {"query_id": "q-e", "relevant_ids": [30], "split": "val"},
    {"query_id": "q-f", "relevant_ids": [30], "split": "val"},
]
GRAPHS = [
    {"vg_image_id": 10, "concept_ids": [1], "triplets": []},
    {"vg_image_id": 20, "concept_ids": [2, 3], "triplets": [[2, 0, 3]]},
    {"vg_image_id": 30, "concept_ids": [], "triplets": []},
]


def test_retrieval_set_is_sorted_by_image_id_with_captions_in_query_id_order():
    images, queries = _tables()
    data, image_ids = build_retrieval_set(Split.TRAIN, IMAGES, QUERIES, GRAPHS, images, queries)
    assert image_ids == (10, 20)
    assert [g.image_id for g in data.graphs] == [10, 20]
    assert [int(g.feature_row[0]) for g in data.graphs] == [1, 2]  # rows of images 10 and 20 in the table
    assert data.caption_vectors.shape == (2, 2, 4)
    expected = torch.from_numpy(queries.select(["q-a", "q-b", "q-c", "q-d"])).reshape(2, 2, 4)
    assert torch.equal(data.caption_vectors, expected)


def test_retrieval_set_rejects_missing_graphs_and_uneven_captions():
    images, queries = _tables()
    with pytest.raises(DataError, match="have no graph"):
        build_retrieval_set(Split.TRAIN, IMAGES, QUERIES, GRAPHS[1:], images, queries)
    with pytest.raises(DataError, match="same, non-zero number of captions"):
        build_retrieval_set(Split.TRAIN, IMAGES, QUERIES[1:], GRAPHS, images, queries)
    with pytest.raises(DataError, match="has no images"):
        build_retrieval_set(Split.TEST, IMAGES, QUERIES, GRAPHS, images, queries)


def test_gradients_are_clipped_to_the_configured_norm(tmp_path, monkeypatch):
    seen = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        seen.append(max_norm)
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(train_module.nn.utils, "clip_grad_norm_", spy)
    _fit(tmp_path, max_epochs=1, grad_clip_norm=0.5)
    assert seen == [0.5, 0.5]  # two batches of 8 images


def test_encoding_and_evaluation_run_in_eval_mode_and_training_in_train_mode():
    model, data, config = _model(), _dataset(n_images=8), _config()
    model.train()
    train_module.encode_graphs(model, data.graphs, config)
    assert not model.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_module.train_epoch(model, optimizer, data, config, seed=0, epoch=1)
    assert model.training


class _FixedVectors(torch.nn.Module):
    """Stand-in encoder: returns a fixed vector per image, looked up by the image row."""

    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return self.vectors[batch.feature_row[batch.ptr[:-1]]]


def test_evaluate_pairs_every_caption_with_the_image_it_belongs_to():
    graph_vectors = torch.eye(8)
    data = _dataset(n_images=8, captions=3)
    # Each image's three captions equal that image's graph vector.
    aligned = RetrievalSet(graphs=data.graphs, caption_vectors=graph_vectors[:, None, :].repeat(1, 3, 1))
    metrics = evaluate(_FixedVectors(graph_vectors), aligned, _config(), KS)
    assert metrics["recall@1"] == 1.0 and metrics["mrr"] == 1.0
    # Captions of image i moved to image i+1: nothing matches any more.
    shifted = RetrievalSet(graphs=data.graphs, caption_vectors=aligned.caption_vectors.roll(1, dims=0))
    assert evaluate(_FixedVectors(graph_vectors), shifted, _config(), KS)["recall@1"] == 0.0


def test_validation_loss_is_the_mean_over_images_of_fixed_first_caption_pairs():
    graph_vectors = _unit((8, DIM), 5)
    captions = _unit((8, 2, DIM), 6)
    data = RetrievalSet(graphs=_dataset(n_images=8).graphs, caption_vectors=captions)
    config = _config(batch_size=4)
    expected = (
        info_nce(graph_vectors[:4], captions[:4, 0], config.temperature).item() * 4
        + info_nce(graph_vectors[4:], captions[4:, 0], config.temperature).item() * 4
    ) / 8
    assert evaluate(_FixedVectors(graph_vectors), data, config, KS)["loss"] == pytest.approx(expected)


def test_epoch_batches_draw_different_captions_and_never_leave_a_single_image_batch():
    batches = epoch_batches(n_images=40, captions_per_image=5, batch_size=8, seed=1, epoch=1)
    assert len({int(c) for _, captions in batches for c in captions}) > 1
    # 9 images in batches of 4 would leave one image alone; it joins the batch before.
    sizes = [len(images) for images, _ in epoch_batches(9, 5, 4, seed=1, epoch=1)]
    assert sizes == [4, 5]
    assert [len(images) for images, _ in epoch_batches(1, 5, 4, seed=1, epoch=1)] == [1]


def test_resume_restores_the_best_epoch_and_stops_where_an_uninterrupted_run_stops(tmp_path):
    # Constant validation metrics: epoch 1 stays best and training stops after epoch 3.
    settings = dict(max_epochs=20, patience=2, learning_rate=1e-12)
    full, _, _ = _fit(tmp_path, "full", **settings)
    _fit(tmp_path, "split", **{**settings, "max_epochs": 2})
    resumed, _, run = _fit(tmp_path, "split", resume=True, **settings)
    assert (
        (resumed.best_epoch, resumed.stop_reason)
        == (full.best_epoch, full.stop_reason)
        == (1, StopReason.NO_IMPROVEMENT)
    )
    assert [row["epoch"] for row in resumed.history] == [1, 2, 3]
    # Resuming a run that already stopped changes nothing.
    again, _, _ = _fit(tmp_path, "split", resume=True, **settings)
    assert _rows(again.history) == _rows(resumed.history) and again.stop_reason is StopReason.NO_IMPROVEMENT
    assert run.load_best()["epoch"] == 1


def test_checkpoint_keeps_the_random_state_and_history_records_the_real_learning_rate(tmp_path):
    result, _, run = _fit(tmp_path, max_epochs=1, learning_rate=0.02)
    assert torch.equal(run.load_last()["rng_state"], torch.get_rng_state())
    assert result.history[0]["learning_rate"] == 0.02


def test_resume_is_rejected_when_best_and_last_checkpoints_disagree(tmp_path):
    _, _, run = _fit(tmp_path, max_epochs=2)
    best = run.load_best()
    run.save_best({**best, "epoch": best["epoch"] + 5})
    with pytest.raises(TrainingError, match="disagree on the best epoch"):
        _fit(tmp_path, resume=True, max_epochs=3)


def test_non_finite_gradient_stops_training_without_updating_the_weights(tmp_path, monkeypatch):
    def clip_that_fails(*args, **kwargs):
        raise RuntimeError("The total norm for gradients is non-finite")

    monkeypatch.setattr(train_module.nn.utils, "clip_grad_norm_", clip_that_fails)
    run = RunDirectory(tmp_path / "run")
    run.prepare(resume=False, overwrite=False)
    model = _model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(TrainingError, match="gradient is not finite at epoch 1"):
        fit(model, _dataset(), _dataset(8, seed=2), _config(), 0, run, KS)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert not run.has_progress()


def test_overwrite_removes_every_file_of_the_earlier_run(tmp_path):
    _, _, run = _fit(tmp_path, max_epochs=1)
    for name in (RESULT_FILE, SETTINGS_FILE, "graph_vectors.npy", "tracker_run_id.txt"):
        (run.path / name).write_text("old")
    (run.path / "wandb").mkdir()
    (run.path / "wandb" / "old-run").write_text("old")
    (run.path / "notes.txt").write_text("mine")
    assert run.has_progress() and run.is_finished()

    run.prepare(resume=False, overwrite=True)
    assert [p.name for p in run.path.iterdir()] == ["notes.txt"]  # only files the run did not write remain
    assert not run.has_progress() and not run.is_finished()
    with pytest.raises(TrainingError, match="cannot resume"):
        run.prepare(resume=True, overwrite=False)
    with pytest.raises(TrainingError, match="not both"):
        run.prepare(resume=True, overwrite=True)


def test_changed_settings_lists_what_would_break_a_resumed_run():
    saved = {
        "graphsage": {"hidden_dim": 512},
        "clip_model": "m",
        "seed": 13,
        "inputs_sha256": {"a": "1"},
        "training": {"batch_size": 256, "learning_rate": 0.001, "max_epochs": 50, "patience": 5, "seeds": [13]},
        "git": {"commit": "abc"},
    }
    assert changed_settings(saved, saved) == []
    longer = {**saved, "training": {**saved["training"], "max_epochs": 80, "patience": 9, "seeds": [13, 37]}}
    assert changed_settings(saved, {**longer, "git": {"commit": "def"}}) == []
    other = {
        **saved,
        "graphsage": {"hidden_dim": 256},
        "inputs_sha256": {"a": "2"},
        "training": {**saved["training"], "learning_rate": 0.5, "temperature": 0.1},
    }
    assert changed_settings(saved, other) == [
        "graphsage",
        "inputs_sha256",
        "training.learning_rate",
        "training.temperature",
    ]


def test_hard_negative_groups_keep_each_image_once_and_put_partners_together():
    partners = [(1, 2, 3), (0,), (0,), (0,), (), (), (), ()]
    batches = epoch_batches(8, 2, 8, seed=1, epoch=0, partners=partners, group_size=4)
    order = batches[0][0].tolist()
    assert sorted(order) == list(range(8))
    start = order.index(0)
    if start + 1 < len(order) and order[start + 1] in (1, 2, 3):
        assert set(order[start:start + 4]) == {0, 1, 2, 3}
