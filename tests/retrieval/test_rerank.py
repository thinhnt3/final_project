import numpy as np
import pytest

from src.core.exceptions import ConfigError, DataError
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData
from src.evaluation.metrics import ranks_of_relevant
from src.retrieval.rerank import clip_candidate_mask, evaluate_reranking, rerank_scores, sweep_alpha_and_k

CLIP = np.array([[0.9, 0.8, 0.7, 0.1], [0.2, 0.6, 0.5, 0.4]])
GRAPH = np.array([[0.0, 0.1, 0.9, 1.0], [0.9, 0.0, 0.8, 0.1]])


def _order(scores):
    return np.argsort(-scores, axis=1).tolist()


def _split(relevant, split=Split.VAL, n_items=4):
    empty = np.zeros((0, 0), dtype=np.float32)
    return SplitData(
        split=split,
        image_ids=tuple(range(n_items)),
        query_ids=tuple(f"q{i}" for i in range(len(relevant))),
        relevant_columns=np.asarray(relevant),
        image_vectors=empty,
        query_vectors=empty,
    )


def test_alpha_one_keeps_the_clip_ranking_for_any_k():
    for top_k in (1, 2, 4):
        assert _order(rerank_scores(CLIP, GRAPH, alpha=1.0, top_k=top_k)) == _order(CLIP)


def test_alpha_one_keeps_ranks_of_nearly_equal_float32_scores():
    # Two float32 scores one step apart. Subtracting the non-candidate offset in float32
    # merges them into a tie, which would push the relevant item from rank 2 to rank 3.
    low = np.float32(0.5)
    high = np.nextafter(low, np.float32(1.0))
    offset = np.float32(3.0)
    assert low < high and low - offset == high - offset
    clip = np.array([[0.9, low, high]], dtype=np.float32)
    relevant = [2]
    assert ranks_of_relevant(clip, relevant).tolist() == [2]
    out = rerank_scores(clip, np.zeros_like(clip), alpha=1.0, top_k=1)
    assert ranks_of_relevant(out, relevant).tolist() == [2]


def test_graph_score_reorders_only_the_top_k_candidates():
    out = rerank_scores(CLIP, GRAPH, alpha=0.0, top_k=3)
    # Query 0: candidates 0, 1, 2 reordered by graph score; item 3 stays last although its
    # graph score is the highest.
    assert _order(out)[0] == [2, 1, 0, 3]
    # Query 1: candidates 1, 2, 3 by graph score; item 0 stays last.
    assert _order(out)[1] == [2, 3, 1, 0]


def test_non_candidates_stay_below_candidates_at_the_extremes_of_the_score_range():
    clip = np.array([[1.0, 0.99, 0.98, 0.97]])
    graph = np.array([[-1.0, -1.0, 1.0, 1.0]])  # worst possible for candidates, best for the rest
    out = rerank_scores(clip, graph, alpha=0.0, top_k=2)
    assert out[0, 2:].max() < out[0, :2].min()
    assert out[0, 2] > out[0, 3]  # non-candidates keep their CLIP order


def test_fused_score_follows_the_formula_inside_the_candidates():
    out = rerank_scores(CLIP, GRAPH, alpha=0.25, top_k=2)
    assert out[0, 0] == pytest.approx(0.25 * 0.9 + 0.75 * 0.0)
    assert out[0, 1] == pytest.approx(0.25 * 0.8 + 0.75 * 0.1)


def test_ties_at_the_cut_are_all_candidates():
    clip = np.array([[0.9, 0.5, 0.5, 0.1]])
    graph = np.array([[0.0, 0.0, 1.0, 1.0]])
    out = rerank_scores(clip, graph, alpha=0.0, top_k=2)
    assert _order(out)[0][0] == 2  # item 2 ties with item 1 at the cut and is reranked


def test_sweep_prefers_recall_at_1_over_mrr():
    # CLIP ranks every relevant item 2nd (R@1 0, MRR 0.5). The graph score ranks one 1st
    # and two last (R@1 1/3, MRR 0.4): better Recall@1, worse MRR.
    clip = np.tile(np.linspace(0.9, 0.0, 10), (3, 1))
    graph = np.zeros((3, 10))
    graph[0, 1], graph[1, 1], graph[2, 1] = 1.0, -1.0, -1.0
    best, points = sweep_alpha_and_k(clip, graph, _split([1, 1, 1], n_items=10), alphas=[1.0, 0.0], top_ks=[10])
    by_alpha = {p.alpha: p.metrics for p in points}
    assert by_alpha[1.0]["recall@1"] == 0.0 and by_alpha[1.0]["mrr"] == pytest.approx(0.5)
    assert by_alpha[0.0]["recall@1"] == pytest.approx(1 / 3) and by_alpha[0.0]["mrr"] == pytest.approx(0.4)
    assert best.alpha == 0.0


def test_sweep_keeps_plain_clip_when_the_graph_score_adds_nothing():
    # A constant graph score cannot change any ranking, so every setting ties.
    best, points = sweep_alpha_and_k(
        CLIP.astype(np.float32), np.zeros_like(CLIP, dtype=np.float32), _split([2, 0]),
        alphas=[0.1, 0.5, 1.0], top_ks=[4, 2],
    )
    assert len({tuple(sorted(p.metrics.items())) for p in points}) == 1
    assert (best.alpha, best.top_k) == (1.0, 2)  # most CLIP weight, then fewest candidates


def test_sweep_reports_recall_at_1_even_when_not_requested():
    best, _ = sweep_alpha_and_k(CLIP, GRAPH, _split([2, 0]), alphas=[0.5], top_ks=[4], ks=(2, 3))
    assert set(best.metrics) == {"recall@1", "recall@2", "recall@3", "mrr"}


def test_sweep_refuses_any_split_but_validation():
    for split in (Split.TEST, Split.TRAIN):
        with pytest.raises(DataError, match="validation only"):
            sweep_alpha_and_k(CLIP, GRAPH, _split([2, 0], split=split), alphas=[0.5], top_ks=[4])


@pytest.mark.parametrize("bad", [5.0, -1.5, np.nan])
def test_scores_outside_the_cosine_range_are_rejected(bad):
    graph = GRAPH.copy()
    graph[0, 0] = bad
    with pytest.raises(DataError, match="graph scores"):
        rerank_scores(CLIP, graph, alpha=0.5, top_k=2)
    with pytest.raises(DataError, match="clip scores"):
        rerank_scores(graph, CLIP, alpha=0.5, top_k=2)


@pytest.mark.parametrize(("alpha", "top_k"), [(1.5, 2), (-0.1, 2), (0.5, 0), (0.5, 5)])
def test_invalid_alpha_or_k_is_rejected(alpha, top_k):
    with pytest.raises(ConfigError):
        rerank_scores(CLIP, GRAPH, alpha=alpha, top_k=top_k)


def test_mismatched_shapes_and_empty_grids_are_rejected():
    with pytest.raises(DataError, match="shapes differ"):
        rerank_scores(CLIP, GRAPH[:, :3], alpha=0.5, top_k=2)
    with pytest.raises(ConfigError, match="must not be empty"):
        sweep_alpha_and_k(CLIP, GRAPH, _split([0, 0]), alphas=[], top_ks=[2])


def test_evaluate_reranking_selects_on_validation_and_applies_to_every_split():
    # Validation: the graph score fixes both queries, so alpha 0 wins. Test: the graph score
    # is wrong, yet the validation-selected alpha must still be applied there.
    val_graph = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    test_graph = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = evaluate_reranking(
        {Split.VAL: CLIP, Split.TEST: CLIP},
        {Split.VAL: val_graph, Split.TEST: test_graph},
        {Split.VAL: _split([2, 0]), Split.TEST: _split([0, 1], split=Split.TEST)},
        alphas=[1.0, 0.0],
        top_ks=[4],
    )
    assert (result.selected.alpha, result.selected.top_k) == (0.0, 4)
    assert result.reranked[Split.VAL]["recall@1"] == 1.0
    assert result.graph_only[Split.VAL]["recall@1"] == 1.0
    assert result.reranked[Split.TEST]["recall@1"] == 0.0  # graph score ranks the wrong items first
    assert len(result.sweep) == 2


def test_evaluate_reranking_needs_validation():
    with pytest.raises(DataError, match="needs the validation split"):
        evaluate_reranking(
            {Split.TEST: CLIP}, {Split.TEST: GRAPH}, {Split.TEST: _split([0, 1], split=Split.TEST)},
            alphas=[1.0], top_ks=[4],
        )


def test_evaluate_reranking_rejects_a_split_without_scores():
    data = {Split.VAL: _split([2, 0]), Split.TEST: _split([0, 1], split=Split.TEST)}
    with pytest.raises(DataError, match="no CLIP or graph scores"):
        evaluate_reranking({Split.VAL: CLIP, Split.TEST: CLIP}, {Split.VAL: GRAPH}, data, alphas=[1.0], top_ks=[4])


def test_evaluate_reranking_rejects_top_k_larger_than_a_gallery_before_sweeping():
    data = {Split.VAL: _split([2, 0]), Split.TEST: _split([0, 1], split=Split.TEST)}
    scores = {Split.VAL: CLIP, Split.TEST: CLIP}
    with pytest.raises(ConfigError, match="exceed the smallest gallery"):
        evaluate_reranking(scores, scores, data, alphas=[1.0], top_ks=[2, 5])


def test_shared_candidate_mask_keeps_ties_at_the_cutoff():
    scores = np.array([[.8, .8, .1]])
    assert clip_candidate_mask(scores, 1).tolist() == [[True, True, False]]
    assert scores.tolist() == [[.8, .8, .1]]
    with pytest.raises(ConfigError):
        clip_candidate_mask(scores, 0)
