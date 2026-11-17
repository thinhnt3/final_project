import pytest

from src.data.review import (
    caption_coverage,
    content_tokens,
    coverage_vs_random,
    describe,
    document_frequency,
    graph_vocabulary,
)


def _graph(*objects):
    return {"objects": [{"label": label, "name_raw": raw} for label, raw in objects]}


def test_content_tokens_drop_stopwords_and_punctuation():
    assert content_tokens("A man riding a bike, on the street.") == ["man", "riding", "bike", "street"]


def test_graph_vocabulary_splits_multiword_labels():
    vocab = graph_vocabulary(_graph(("gym shoe", "sneakers"), ("bicycle", "bikes")))
    assert vocab == {"gym", "shoe", "sneakers", "bicycle", "bikes"}


def test_caption_coverage_handles_plurals():
    vocab = graph_vocabulary(_graph(("man", "man"), ("bicycle", "bikes")))
    # content words: man, riding, bike → man and bike (via `bikes`) match
    assert caption_coverage("A man riding a bike", vocab) == pytest.approx(2 / 3)
    assert caption_coverage("the of a", vocab) is None


def test_coverage_vs_random_prefers_own_graph():
    captions = {1: ["a dog on grass"], 2: ["a red car on a road"]}
    vocab = {1: {"dog", "grass"}, 2: {"car", "road", "red"}}
    stats = coverage_vs_random(captions, vocab, seed=0)
    assert stats.captions == 2
    assert stats.own_mean == 1.0
    assert stats.random_mean == 0.0
    assert stats.own_any_match_pct == 100.0


def test_describe_and_document_frequency():
    summary = describe(list(range(1, 11)))
    assert summary["median"] == 5.5 and summary["max"] == 10
    assert document_frequency([["wall", "wall", "sky"], ["wall"]]) == {"wall": 2, "sky": 1}
