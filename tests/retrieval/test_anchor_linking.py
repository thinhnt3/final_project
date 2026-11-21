import pytest

from src.core.exceptions import DataError
from src.retrieval.anchor_linking import AnchorLinker

NAMES = ["Film", "Doggfather", "Snoop Dogg", "The Abyss", "Abyss"]


def test_finds_a_name_inside_a_question():
    assert AnchorLinker(NAMES).link("Who performed Doggfather?") == [1]


def test_matching_is_case_sensitive():
    assert AnchorLinker(NAMES).link("a film written by somebody") == []


def test_longest_name_wins_and_order_follows_the_text():
    assert AnchorLinker(NAMES).link("Snoop Dogg's favourite film is The Abyss") == [2, 3]


def test_duplicate_names_are_rejected():
    with pytest.raises(DataError):
        AnchorLinker(["Film", "Film"])
