import pytest

from src.core.exceptions import DataError
from src.data.mkg_w_config import QueryRules, RelationTemplate
from src.data.mkg_w_handwritten import resolve_handwritten
from src.data.mkg_w_queries import QueryKind, build_chain_queries, direction_stats
from src.data.mkg_w_raw import Triple
from src.data.schemas import Split

RULES = QueryRules(prior_top_k=1, max_prior_hits=0.9, min_train_triples=1, max_answers=5, max_hops=3)
DIRECTOR, FOLLOWS = 0, 1
# Films 0-3, directors 4-5. Film 1 follows film 0 (train); film 1 is directed by 4 (test).
TRIPLES = {
    Split.TRAIN: [Triple(1, FOLLOWS, 0), Triple(0, DIRECTOR, 5), Triple(2, DIRECTOR, 4), Triple(3, FOLLOWS, 2)],
    Split.VAL: [],
    Split.TEST: [Triple(1, DIRECTOR, 4)],
}
NAMES = ["Film_0", "Film_1", "Film_2", "Film_3", "Director_4", "Director_5"]
TEMPLATES = {
    DIRECTOR: RelationTemplate("P57", "director", "the director of {anchor}", "a film directed by {anchor}"),
    FOLLOWS: RelationTemplate("P155", "follows", "the film before {anchor}", "the film after {anchor}"),
}
ENTITY_OF = {name.replace("_", " "): i for i, name in enumerate(NAMES)}
RELATION_OF = {"P57": DIRECTOR, "P155": FOLLOWS}


def test_chain_query_joins_a_train_fact_with_a_test_fact():
    stats = direction_stats(TRIPLES[Split.TRAIN], RULES)
    queries = build_chain_queries(TRIPLES, Split.TEST, stats, TEMPLATES, NAMES, [True] * 6, RULES.max_answers)
    by_text = {q.text: q for q in queries}
    query = by_text["the director of the film after Film 0"]
    assert (query.kind, query.anchor_id, query.relevant_ids) == (QueryKind.CHAIN, 0, (4,))
    # Film 2 is also directed by 4, and film 3 comes after it: the same test fact read backwards.
    assert by_text["a film directed by the director of Film 2"].relevant_ids == (1,)
    assert all(q.anchor_id not in q.relevant_ids for q in queries)


def _item(**changes):
    item = {"text": "Who directed the sequel to Film 0?", "kind": "chain", "answers": ["Director 4"],
            "steps": [{"anchor": "Film 0", "relation": "P155", "answer_is": "head"},
                      {"relation": "P57", "answer_is": "tail"}]}
    return {**item, **changes}


def test_handwritten_chain_is_checked_and_resolved():
    record = resolve_handwritten([_item()], ENTITY_OF, RELATION_OF, TRIPLES, [True] * 6)[0]
    assert (record["anchor_ids"], record["relevant_ids"], record["kind"]) == ([0], [4], "handwritten_chain")


@pytest.mark.parametrize("changes", [
    {"answers": ["Director 5"]},  # does not satisfy the steps
    {"answers": ["Nobody"]},  # unknown entity
    {"kind": "single"},  # two steps do not fit one fact
    {"kind": "intersection", "answers": ["Director 4"],
     "steps": [{"anchor": "Film 2", "relation": "P57", "answer_is": "tail"},
               {"anchor": "Film 2", "relation": "P57", "answer_is": "tail"}]},  # already in the train graph
])
def test_handwritten_errors(changes):
    with pytest.raises(DataError):
        resolve_handwritten([_item(**changes)], ENTITY_OF, RELATION_OF, TRIPLES, [True] * 6)


def test_handwritten_answer_needs_an_image():
    with pytest.raises(DataError):
        resolve_handwritten([_item()], ENTITY_OF, RELATION_OF, TRIPLES, [True] * 4 + [False, True])
