from pathlib import Path

from src.data.mkg_w_captions import (
    CENTER_CONCEPT,
    CaptionSubsetConfig,
    build_vocabulary,
    item_graph,
    item_triples,
    masked_captions,
    select_items,
)
from src.data.mkg_w_raw import Triple, split_sentences
from src.data.schemas import Split

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "mkg_w_captions.yaml"
CONFIG = CaptionSubsetConfig(Path("unused.yaml"), min_triples=2, captions_per_item=2, min_caption_chars=20,
                             mask_text="this entity", fractions={Split.TRAIN: 0.5, Split.VAL: 0.25, Split.TEST: 0.25})
DIRECTOR, CAST = 0, 1
# Film 0: directed by 2, starring 3. Film 1: directed by 2 only.
TRIPLES = [Triple(0, DIRECTOR, 2), Triple(0, CAST, 3), Triple(1, DIRECTOR, 2)]
NAMES = ["Signs_(film)", "Other_Film", "A_Director", "An_Actor"]
DESCRIPTION = ("Signs (film) is a 2002 film directed by A Director. "
               "Signs stars An Actor in the leading role of the story. Short.")


def _entity(entity_id: int, images: bool = True) -> dict:
    return {"entity_id": entity_id, "name": NAMES[entity_id], "description": DESCRIPTION,
            "image_files": ["0.jpg"] if images else []}


def test_repo_config_loads():
    assert CaptionSubsetConfig.from_yaml(REPO_CONFIG).min_triples == 5


def test_split_sentences_keeps_name_initials_together():
    text = "A 1955 western starring Glenn Ford and Edward G. Robinson in the lead. It was shot in colour in Arizona."
    assert len(split_sentences(text)) == 2


def test_captions_mask_the_name_with_and_without_its_parenthesis():
    captions = masked_captions(DESCRIPTION, NAMES[0], CONFIG)
    assert captions == ["This entity is a 2002 film directed by A Director.",
                        "This entity stars An Actor in the leading role of the story."]


def test_select_items_needs_image_and_enough_triples():
    items = select_items([_entity(0), _entity(1), _entity(2, images=False)], TRIPLES, CONFIG)
    assert list(items) == [0]  # film 1 has one triple; the director has no image


def test_item_graph_replaces_the_item_by_the_center_concept():
    touching = item_triples(TRIPLES)
    split = {0: Split.TRAIN, 1: Split.TEST}
    concepts, predicates, concept_of, predicate_of = build_vocabulary(
        split, touching, NAMES, ["director", "cast member"], CONFIG.mask_text)
    assert concepts[CENTER_CONCEPT]["text"] == "this entity"
    assert {c["text"] for c in concepts[1:]} == {"A Director", "An Actor"}
    graph = item_graph(0, Split.TRAIN, touching[0], concept_of, predicate_of)
    assert sorted(graph["triplets"]) == sorted([
        [CENTER_CONCEPT, predicate_of[DIRECTOR], concept_of[2]],
        [CENTER_CONCEPT, predicate_of[CAST], concept_of[3]],
    ])
    assert "Signs" not in " ".join(concepts[c]["text"] for c in graph["concept_ids"])
    assert [p["train_images"] for p in predicates] == [1, 1]
