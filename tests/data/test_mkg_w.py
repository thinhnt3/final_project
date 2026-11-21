import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.core.exceptions import ConfigError, DataError
from src.data.mkg_w_config import MkgwConfig, QueryRules, RelationTemplate
from src.data.mkg_w_images import extract_images, image_members
from src.data.mkg_w_queries import (
    Direction,
    DirectionStatus,
    QueryKind,
    anchor_answer_hops,
    build_direct_queries,
    build_indirect_queries,
    direction_stats,
    split_entities,
)
from src.data.mkg_w_raw import (
    Triple,
    first_sentence,
    match_folders,
    read_descriptions,
    read_id_file,
    read_triples,
)
from src.data.schemas import Split

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "mkg_w.yaml"
RULES = QueryRules(prior_top_k=1, max_prior_hits=0.5, min_train_triples=4, max_answers=2, max_hops=3)

# Relation 0 "director": films 0-3 → directors 10-13 (answers all different).
# Relation 1 "instance of": films 0-3 → class 20 (one answer for every triple).
DIRECTOR, INSTANCE_OF = 0, 1
TRAIN = [Triple(film, DIRECTOR, 10 + film) for film in range(4)] + [Triple(film, INSTANCE_OF, 20) for film in range(4)]
NAMES = [f"Film_{i}" for i in range(10)] + [f"Director_{i}" for i in range(10)] + ["Film"]
TEMPLATES = {
    DIRECTOR: RelationTemplate("P57", "director", "the director of {anchor}", "a film directed by {anchor}"),
    INSTANCE_OF: RelationTemplate("P31", "instance of", None, "an example of {anchor}"),
}


def test_repo_config_loads():
    config = MkgwConfig.from_yaml(REPO_CONFIG)
    assert config.raw_path("mkg_w_images") == Path("data/raw/mkg_w/MKG-W_img.zip")
    assert config.templates["P57"].tail == "the director of {anchor}"
    assert config.templates["P31"].tail is None


def test_template_needs_one_anchor():
    raw = {
        "paths": {k: "x" for k in ("raw_dir", "manifest_file", "processed_dir", "split_file", "card_file")},
        "sources": [{"name": "a", "url": "file:///a", "target": "a"}],
        "query_rules": {"prior_top_k": 10, "max_prior_hits": 0.3, "min_train_triples": 200, "max_answers": 10,
                        "max_hops": 3},
        "direct_split": {"seed": 1, "fractions": {"train": 0.7, "val": 0.15, "test": 0.15}},
        "templates": {"P57": {"label": "director", "tail": "the director"}},
    }
    with pytest.raises(ConfigError, match="anchor"):
        MkgwConfig.from_dict(raw)


def test_read_triples_uses_head_tail_relation_order(tmp_path):
    path = tmp_path / "train2id.txt"
    path.write_text("2\n0 1 0\n2 3 1\n")
    assert read_triples(path, n_entities=4, n_relations=2) == [Triple(0, 0, 1), Triple(2, 1, 3)]
    path.write_text("1\n0 9 0\n")
    with pytest.raises(DataError, match="out of range"):
        read_triples(path, n_entities=4, n_relations=2)


def test_read_id_file_orders_by_id(tmp_path):
    path = tmp_path / "entity2id.txt"
    path.write_text("2\nhttp://www.wikidata.org/entity/Q2 1\nhttp://www.wikidata.org/entity/Q1 0\n")
    assert read_id_file(path) == ["Q1", "Q2"]


def test_read_descriptions_strips_literal_markup(tmp_path):
    path = tmp_path / "desc.txt"
    path.write_text('http://dbpedia.org/resource/Allan_Dwan\t"Allan Dwan was a \\"pioneering\\" director."@en .\n')
    assert read_descriptions(path) == {"Allan_Dwan": 'Allan Dwan was a "pioneering" director.'}


def test_first_sentence_skips_abbreviations_and_caps_length():
    text = "St. Mary Hall is a boarding school founded in 1850 in the city. It has 300 pupils."
    assert first_sentence(text) == "St. Mary Hall is a boarding school founded in 1850 in the city."
    assert len(first_sentence("word " * 200, max_chars=50)) <= 50


def test_match_folders_exact_then_unique_loose():
    names = {0: "Like_Mike", 1: "Madonna:_Truth_or_Dare", 2: "A.B", 3: "A-B", 4: "No_Folder"}
    folders = ["Like_Mike", "Madonna__Truth_or_Dare", "AB", "Stray"]
    # 2 and 3 share the loose key "ab", so neither takes the folder "AB".
    assert match_folders(names, folders) == {0: "Like_Mike", 1: "Madonna__Truth_or_Dare"}


def test_direction_rules():
    stats = direction_stats(TRAIN, RULES)
    assert stats[(DIRECTOR, Direction.TAIL)].status is DirectionStatus.KEPT
    assert stats[(DIRECTOR, Direction.HEAD)].status is DirectionStatus.KEPT
    # Guessing "Film" answers every instance-of triple, so the tail direction is dropped.
    assert stats[(INSTANCE_OF, Direction.TAIL)].prior_hits == 1.0
    assert stats[(INSTANCE_OF, Direction.TAIL)].status is DirectionStatus.ANSWERS_ARE_CLASSES
    assert stats[(INSTANCE_OF, Direction.HEAD)].status is DirectionStatus.KEPT
    small = direction_stats(TRAIN[:2], RULES)
    assert small[(DIRECTOR, Direction.TAIL)].status is DirectionStatus.TOO_FEW_TRIPLES


def _indirect(test_triples, has_image=None):
    triples = {Split.TRAIN: TRAIN, Split.VAL: [], Split.TEST: test_triples}
    return build_indirect_queries(
        triples, direction_stats(TRAIN, RULES), TEMPLATES, NAMES,
        has_image or [True] * len(NAMES), RULES.max_answers,
    )


def test_indirect_query_text_and_filter_set():
    # Film 0 gets a second director in test: train answer 10 is known, test answer 15 is relevant.
    queries = {q.query_id: q for q in _indirect([Triple(0, DIRECTOR, 15)])}
    query = queries[f"rel-0-{DIRECTOR}-tail-test"]
    assert query.text == "the director of Film 0"
    assert query.relevant_ids == (15,) and query.known_ids == (10,)
    assert (query.anchor_id, query.relation_id, query.direction) == (0, DIRECTOR, Direction.TAIL)
    assert queries[f"rel-15-{DIRECTOR}-head-test"].text == "a film directed by Director 5"


def test_anchor_with_too_many_answers_is_dropped():
    # "an example of Film" has 4 answers > max_answers=2, so no query names the class as anchor.
    assert not [q for q in _indirect([]) if q.relation_id == INSTANCE_OF]


def test_answer_without_image_is_not_relevant():
    has_image = [True] * len(NAMES)
    has_image[15] = False
    queries = _indirect([Triple(0, DIRECTOR, 15)], has_image)
    assert f"rel-0-{DIRECTOR}-tail-test" not in {q.query_id for q in queries}
    # The train query of the same anchor still filters the imageless answer.
    train_query = next(q for q in queries if q.query_id == f"rel-0-{DIRECTOR}-tail-train")
    assert train_query.known_ids == (15,)


def test_missing_template_fails():
    triples = {Split.TRAIN: TRAIN, Split.VAL: [], Split.TEST: []}
    with pytest.raises(DataError, match="without a template"):
        build_indirect_queries(triples, direction_stats(TRAIN, RULES), {DIRECTOR: TEMPLATES[DIRECTOR]},
                               NAMES, [True] * len(NAMES), RULES.max_answers)


def test_anchor_answer_hops():
    writer, producer = 2, 3
    train = [
        *TRAIN,
        Triple(0, writer, 10),    # film 0's director also wrote it
        Triple(1, producer, 12),  # film 1 → director 12 ← film 2
    ]
    triples = {Split.TRAIN: train, Split.VAL: [], Split.TEST: [Triple(1, DIRECTOR, 12), Triple(3, DIRECTOR, 19)]}
    queries = build_indirect_queries(triples, direction_stats(train, RULES), TEMPLATES, NAMES,
                                     [True] * len(NAMES), max_answers=5)
    hops = dict(zip((q.query_id for q in queries), anchor_answer_hops(queries, train, max_hops=3), strict=True))
    # Train query: the asked fact is in the graph, but the writer edge links the same pair.
    assert hops[f"rel-0-{DIRECTOR}-tail-train"] == 1
    # Train query linked only by the asked fact: next best path is film 3 → Film → film 0 → ... none to 13.
    assert hops[f"rel-3-{DIRECTOR}-tail-train"] is None
    # Test fact is absent from train; film 1 reaches director 12 through the producer edge.
    assert hops[f"rel-1-{DIRECTOR}-tail-test"] == 1
    # Director 19 has no train edge at all.
    assert hops[f"rel-3-{DIRECTOR}-tail-test"] is None
    # Film 2 → director 12 is asked in train; the other route is film 2 → Film → film 1 → 12.
    assert hops[f"rel-2-{DIRECTOR}-tail-train"] == 3


def test_split_entities_is_a_seeded_partition():
    fractions = {Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15}
    first = split_entities(list(range(100)), fractions, seed=13)
    assert [len(first[s]) for s in Split] == [70, 15, 15]
    assert sorted(i for ids in first.values() for i in ids) == list(range(100))
    assert first == split_entities(list(reversed(range(100))), fractions, seed=13)
    assert first != split_entities(list(range(100)), fractions, seed=14)


def test_direct_queries():
    queries = build_direct_queries({Split.TEST: [0, 10]}, NAMES, {10: "Director 0 is a director. More text."})
    assert [(q.kind, q.text, q.relevant_ids) for q in queries] == [
        (QueryKind.NAME, "Film 0", (0,)),
        (QueryKind.NAME, "Director 0", (10,)),
        (QueryKind.DESCRIPTION, "Director 0 is a director. More text.", (10,)),
    ]


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_images_skips_files_that_do_not_decode(tmp_path):
    archive_path = tmp_path / "img.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("root/Like_Mike/1.poster.png", _png())
        zf.writestr("root/Like_Mike/2.broken.jpg", b"not an image")
        zf.writestr("root/Like_Mike/Thumbs.db", b"x")
        zf.writestr("root/Only_Broken/1.jpg", b"not an image")
    with zipfile.ZipFile(archive_path) as archive:
        members = image_members(archive)
        assert {k: len(v) for k, v in members.items()} == {"Like_Mike": 2, "Only_Broken": 1}
        files, unreadable = extract_images(archive, {7: "Like_Mike", 8: "Only_Broken"}, members, tmp_path / "out")
    assert files == {7: ["7/0.png"]}
    assert sorted(unreadable) == ["root/Like_Mike/2.broken.jpg", "root/Only_Broken/1.jpg"]
    assert (tmp_path / "out" / "7" / "0.png").read_bytes() == _png()
