from src.data.text_match import names, names_any
from src.models.train_data import relation_mentions


def test_names_matches_whole_words_and_plurals():
    assert names("Two horses next to a man", "horse")
    assert names("a cat sitting next to a dog", "next to")
    assert not names("a seahorse toy", "horse")
    assert names_any("a boy riding", ["man", "boy"])


def test_relation_mentions_point_at_named_relation_nodes():
    graphs = [{"vg_image_id": 1, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2]]}]
    queries = [
        {"query_id": "a", "relevant_ids": [1], "text": "A man riding a horse."},
        {"query_id": "b", "relevant_ids": [1], "text": "A man and his dog."},
    ]
    concepts = {0: ("man",), 1: ("horse",), 2: ("dog",)}
    mentions = relation_mentions(graphs, [1], queries, concepts, {0: "riding", 1: "near"})
    # Nodes: image 0, objects 1..3, relations 4 (man riding horse) and 5 (man near dog).
    assert mentions == (((4,), ()),)
