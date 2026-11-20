import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.evaluate_adaptive_fusion as script
from src.core.exceptions import ConfigError, DataError
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData
from src.evaluation.metrics import ranks_of_relevant
from src.retrieval.adaptive_fusion import (
    AdaptiveFusion,
    AdaptiveFusionConfig,
    BucketFeature,
    BucketSetting,
    adaptive_scores,
    assign_buckets,
    bucket_edges,
    clip_margin,
    fit_adaptive_fusion,
    random_feature,
)
from src.retrieval.config import RetrievalConfig
from src.retrieval.rerank import rerank_scores, sweep_alpha_and_k

ALPHAS = tuple(round(i / 100, 2) for i in range(101))

# Two kinds of query, three images each. "Unsure": CLIP puts a wrong image 0.02 above the right
# one (column 1) and the graph weakly favours the right one, so it needs alpha < 0.834.
# "Sure": CLIP is right (column 0) by 0.05 and the graph strongly favours the wrong image, so it
# needs alpha > 0.952. No single alpha serves both; one per bucket does.
UNSURE_CLIP, UNSURE_GRAPH = [0.30, 0.28, 0.0], [0.0, 0.1, 0.0]
SURE_CLIP, SURE_GRAPH = [0.60, 0.55, 0.0], [0.0, 1.0, 0.0]


def _split(relevant, split=Split.VAL, n_items=3, prefix="q"):
    return SplitData(
        split=split,
        image_ids=tuple(range(n_items)),
        query_ids=tuple(f"{prefix}{i}" for i in range(len(relevant))),
        relevant_columns=np.asarray(relevant),
        image_vectors=np.zeros((n_items, 1), dtype=np.float32),
        query_vectors=np.zeros((len(relevant), 1), dtype=np.float32),
    )


def _conflicting(n_each=4):
    clip = np.array([UNSURE_CLIP] * n_each + [SURE_CLIP] * n_each)
    graph = np.array([UNSURE_GRAPH] * n_each + [SURE_GRAPH] * n_each)
    return clip, graph, [1] * n_each + [0] * n_each


def _fit(clip, graph, data, n_buckets, feature=BucketFeature.CLIP_MARGIN, top_ks=(2, 3), min_bucket_queries=1):
    return fit_adaptive_fusion(
        clip, graph, data, feature, n_buckets, ALPHAS, top_ks, min_bucket_queries=min_bucket_queries, feature_seed=0
    )


def test_clip_margin_is_top_one_minus_top_two():
    scores = np.array([[0.9, 0.5, 0.7], [0.2, 0.2, 0.1]])
    np.testing.assert_allclose(clip_margin(scores), [0.2, 0.0])


def test_clip_margin_needs_two_items():
    with pytest.raises(DataError):
        clip_margin(np.array([[0.5], [0.4]]))


def test_random_feature_depends_only_on_seed_and_query_id():
    first = random_feature(["q1", "q2", "q3"], seed=7)
    shuffled = random_feature(["q3", "q1", "q2"], seed=7)
    assert len(set(first.tolist())) == 3  # different queries get different values
    assert (first[0], first[1], first[2]) == (shuffled[1], shuffled[2], shuffled[0])
    assert not np.array_equal(first, random_feature(["q1", "q2", "q3"], seed=8))
    assert ((first >= 0) & (first < 1)).all()


def test_bucket_edges_are_quantiles_and_tied_quantiles_collapse():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert bucket_edges(values, 1) == ()
    assert bucket_edges(values, 2) == (2.5,)
    np.testing.assert_allclose(bucket_edges(values, 4), [1.75, 2.5, 3.25])
    assert bucket_edges(np.full(6, 5.0), 3) == (5.0,)
    with pytest.raises(ConfigError):
        bucket_edges(values, 0)


def test_a_value_on_an_edge_goes_to_the_lower_bucket():
    np.testing.assert_array_equal(assign_buckets(np.array([1.0, 2.5, 2.6, 9.0]), (2.5,)), [0, 0, 1, 1])
    np.testing.assert_array_equal(assign_buckets(np.array([3.0, 4.0]), ()), [0, 0])


def test_one_bucket_reproduces_the_official_selection_and_scores():
    rng = np.random.default_rng(0)
    clip, graph = rng.uniform(-1, 1, (30, 8)), rng.uniform(-1, 1, (30, 8))
    data = _split(rng.integers(0, 8, 30), n_items=8)
    fusion = _fit(clip, graph, data, 1, top_ks=(2, 4, 8))
    best, _ = sweep_alpha_and_k(clip, graph, data, ALPHAS, (2, 4, 8))
    (setting,) = fusion.settings
    assert (setting.alpha, setting.top_k) == (best.alpha, best.top_k)
    fused, buckets = adaptive_scores(fusion, clip, graph, data.query_ids)
    np.testing.assert_array_equal(fused, rerank_scores(clip, graph, best.alpha, best.top_k))
    assert not buckets.any()


def test_each_bucket_gets_its_own_setting_where_one_alpha_cannot_serve_both():
    clip, graph, relevant = _conflicting()
    data = _split(relevant)
    fixed = _fit(clip, graph, data, 1)
    assert fixed.settings[0].val_metrics["recall@1"] == 0.5
    adaptive = _fit(clip, graph, data, 2)
    unsure, sure = adaptive.settings
    assert unsure.alpha <= 0.83 and sure.alpha >= 0.96
    fused, buckets = adaptive_scores(adaptive, clip, graph, data.query_ids)
    np.testing.assert_array_equal(buckets, [0] * 4 + [1] * 4)
    assert (ranks_of_relevant(fused, data.relevant_columns) == 1).all()


def test_fitting_refuses_any_split_but_validation():
    clip, graph, relevant = _conflicting()
    with pytest.raises(DataError):
        _fit(clip, graph, _split(relevant, split=Split.TEST), 2)


def test_buckets_too_small_to_select_on_are_refused():
    clip, graph, relevant = _conflicting(n_each=2)
    with pytest.raises(ConfigError):
        _fit(clip, graph, _split(relevant), 2, min_bucket_queries=3)
    # Tied features collapse the edges and leave an empty bucket, which is refused too.
    tied = np.array([UNSURE_CLIP] * 4)
    with pytest.raises(ConfigError):
        _fit(tied, np.array([UNSURE_GRAPH] * 4), _split([1] * 4), 2)


def test_scoring_another_split_uses_the_validation_edges():
    clip, graph, relevant = _conflicting()
    fusion = _fit(clip, graph, _split(relevant), 2)
    assert fusion.edges == pytest.approx((0.035,))
    # Margins 0.01, 0.02, 0.03 all sit below the validation edge. Edges recomputed on these
    # queries (median 0.02) would move the last one into the upper bucket.
    test_clip = np.array([[0.30, 0.29, 0.0], [0.30, 0.28, 0.0], [0.30, 0.27, 0.0]])
    _, buckets = adaptive_scores(fusion, test_clip, np.zeros_like(test_clip), ("t0", "t1", "t2"))
    np.testing.assert_array_equal(buckets, [0, 0, 0])


def test_alpha_one_in_every_bucket_keeps_the_clip_ranking():
    rng = np.random.default_rng(1)
    clip, graph = rng.uniform(-1, 1, (20, 6)), rng.uniform(-1, 1, (20, 6))
    settings = (BucketSetting(1.0, 2, 1, {}), BucketSetting(1.0, 6, 1, {}))
    fusion = AdaptiveFusion(BucketFeature.CLIP_MARGIN, 0, (float(np.median(clip_margin(clip))),), settings)
    fused, buckets = adaptive_scores(fusion, clip, graph, tuple(f"q{i}" for i in range(20)))
    assert set(buckets.tolist()) == {0, 1}
    np.testing.assert_array_equal(np.argsort(-fused, axis=1), np.argsort(-clip, axis=1))


def test_a_fitted_fusion_needs_one_setting_per_bucket():
    with pytest.raises(DataError):
        AdaptiveFusion(BucketFeature.CLIP_MARGIN, 0, (0.1,), (BucketSetting(0.5, 2, 1, {}),))


def _raw_config(**overrides):
    raw = {
        "retrieval_config": "configs/retrieval/ablations/full.yaml",
        "features": ["clip_margin", "random"],
        "n_buckets": [1, 2, 4],
        "min_bucket_queries": 500,
        "feature_seed": 2026,
        "bootstrap_repeats": 2000,
        "bootstrap_seed": 310,
        "output": "runs/analysis/adaptive_fusion.json",
    }
    return {**raw, **overrides}


def test_config_requires_the_fixed_baseline():
    with pytest.raises(ConfigError):
        AdaptiveFusionConfig.from_dict(_raw_config(n_buckets=[2, 4]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"features": ["clip_margin", "magic"]},
        {"features": ["random", "random"]},
        {"features": []},
        {"n_buckets": [1, 2, 2]},
        {"n_buckets": [1, True]},
        {"n_buckets": 3},
        {"n_buckets": [1, 0]},
        {"min_bucket_queries": 0},
        {"bootstrap_repeats": 0},
        {"output": 5},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ConfigError):
        AdaptiveFusionConfig.from_dict(_raw_config(**overrides))


def test_shipped_config_is_valid():
    config = AdaptiveFusionConfig.from_yaml("configs/evaluation/adaptive_fusion.yaml")
    assert 1 in config.n_buckets
    assert config.features == (BucketFeature.CLIP_MARGIN, BucketFeature.RANDOM)


def _script_world(monkeypatch, tmp_path):
    """The script on the conflicting queries: real config parsing and orchestration, but the
    vector loading and the provenance checks of real artifacts are faked. Returns the fake
    benchmark (with the official selection of every seed) and the output path."""
    shipped = RetrievalConfig.from_yaml("configs/retrieval/ablations/full.yaml")
    retrieval = replace(shipped, benchmark=replace(shipped.benchmark, alpha_grid=ALPHAS[::10], top_k_grid=(2, 3)))
    grid = retrieval.benchmark
    data, clip, graph = {}, {}, {}
    for split, n_each in ((Split.VAL, 20), (Split.TEST, 15)):
        clip[split], graph[split], relevant = _conflicting(n_each)
        data[split] = _split(relevant, split=split, prefix=split.value)
    best, _ = sweep_alpha_and_k(clip[Split.VAL], graph[Split.VAL], data[Split.VAL], grid.alpha_grid, grid.top_k_grid)
    official = {"alpha": best.alpha, "top_k": best.top_k}
    benchmark = {"seeds": {str(seed): {"selected_on_validation": dict(official)} for seed in retrieval.training.seeds}}
    fakes = {
        "RetrievalConfig": SimpleNamespace(from_yaml=lambda path: retrieval),
        "verify_inputs_unchanged": lambda directory: None,
        "read_json": lambda path: benchmark if Path(path).parent.name == "benchmark" else [],
        "load_table": lambda directory, name: None,
        "build_split_data": lambda split, *_: data[split],
        "clip_scores": lambda split_data: clip[split_data.split],
        "graph_vector_scores": lambda split_data, table: graph[split_data.split],
        "validate_artifact_inputs": lambda *_: None,
        "validate_run": lambda *_: None,
        "sha256_of": lambda path: "fake",
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(script, name, fake)
    config, output = tmp_path / "adaptive_fusion.yaml", tmp_path / "adaptive_fusion.json"
    config.write_text(
        "retrieval_config: configs/retrieval/ablations/full.yaml\nfeatures: [clip_margin, random]\n"
        "n_buckets: [1, 2]\nmin_bucket_queries: 5\nfeature_seed: 0\nbootstrap_repeats: 50\n"
        f"bootstrap_seed: 0\noutput: {output}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_adaptive_fusion", "--config", str(config)])
    return benchmark, output


def test_script_compares_every_bucketing_with_the_fixed_baseline(monkeypatch, tmp_path):
    _, output = _script_world(monkeypatch, tmp_path)
    assert script.main() == 0
    results = json.loads(output.read_text(encoding="utf-8"))["results"]
    assert set(results) == {"fixed", "clip_margin-b2", "random-b2"}
    assert "paired_vs_fixed_test" not in results["fixed"]
    margin = results["clip_margin-b2"]
    assert results["fixed"]["summary"]["test"]["recall@1"]["mean"] == 0.5
    assert margin["summary"]["test"]["recall@1"]["mean"] == 1.0
    assert margin["paired_vs_fixed_test"]["delta_r1_pp"] == pytest.approx(50.0)
    for seed_result in margin["seeds"].values():
        buckets = seed_result["test"]["buckets"]
        assert [b["queries"] for b in buckets] == [15, 15]
        assert [b["recall@1"] for b in buckets] == [1.0, 1.0]
        # The fixed alpha serves one kind of query and fails the other.
        assert sorted(b["fixed_recall@1"] for b in buckets) == [0.0, 1.0]


def test_script_stops_when_one_bucket_differs_from_the_benchmark(monkeypatch, tmp_path):
    benchmark, output = _script_world(monkeypatch, tmp_path)
    official = next(iter(benchmark["seeds"].values()))["selected_on_validation"]
    official["top_k"] = 3 if official["top_k"] == 2 else 2
    assert script.main() == 1
    assert not output.exists()
