import pytest

from src.core.exceptions import DataError
from src.data.synsets import normalize_predicate, normalize_text, synset_lemma


@pytest.mark.parametrize("synset, lemma", [
    ("bicycle.n.01", "bicycle"),
    ("gym_shoe.n.01", "gym shoe"),
    ("man.n.01", "man"),
    ("t.v..n.01", "t.v."),  # lemma containing dots
])
def test_synset_lemma(synset, lemma):
    assert synset_lemma(synset) == lemma


@pytest.mark.parametrize("bad", ["", "bicycle", "bicycle.n", ".n.01", "bicycle.n.x"])
def test_synset_lemma_rejects_malformed(bad):
    with pytest.raises(DataError):
        synset_lemma(bad)


@pytest.mark.parametrize("raw, expected", [
    (" ON ", "on"),
    ("Next  To", "next to"),
    ("wears.", "wears"),
    ("in front of", "in front of"),
    ("...", ""),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("on a", "on"),
    ("has a", "has"),
    ("Wearing The", "wearing"),
    ("are on", "on"),
    ("is on a", "on"),
    ("in front of", "in front of"),
    ("a", ""),                      # nothing left → dropped later as empty
    ("is", ""),
    ("wears", "wears"),             # inflections are not merged
    ("a part of", "a part of"),     # only trailing articles are removed
])
def test_normalize_predicate(raw, expected):
    assert normalize_predicate(raw) == expected
