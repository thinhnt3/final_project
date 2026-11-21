from src.data.vn_heritage_articles import (
    Cue,
    LinkRecord,
    Mention,
    RelationScheme,
    body_link_records,
    collect_mentions,
    cue_relation,
    follows_any,
    mention_relation,
    normalise_title,
    strip_templates,
    unresolved_shared_titles,
)

STOP = frozenset({"xem thêm", "tham khảo"})
BUCKETS = {"lịch sử": "history", "kiến trúc": "architecture"}
CUES = (
    Cue("built_by", ("cho xây", "xây dựng bởi"), frozenset({"person"})),
    Cue("dated_to_period", ("thời", "triều"), frozenset({"person", "era"})),
    Cue("worships", ("thờ",), frozenset({"person", "deity"})),
)


def _record(title: str, before: str = "", after: str = "", section: str = "lead") -> LinkRecord:
    return LinkRecord(title, section, before, after)


def test_strip_templates_handles_nesting_and_stray_braces() -> None:
    assert strip_templates("a {{x|{{y}}|[[Navbox link]]}} b }} c") == "a  b }} c"


def test_normalise_title() -> None:
    assert normalise_title(" nhà_Nguyễn  ") == "Nhà Nguyễn"


def test_body_link_records_skip_templates_refs_namespaces_and_reference_sections() -> None:
    wikitext = (
        "{{Infobox|vị trí=[[Huế]]}}\n"
        "'''Chùa A''' ở [[hà Nội|thủ đô]], gần [[Hồ_Tây#Lịch sử]].<ref>[[Nguồn]]</ref>\n"
        "[[Thể loại:Chùa]] <!-- [[Ẩn]] -->\n"
        "== Xem thêm ==\n* [[Chùa B]]\n"
        "{{Chùa Việt Nam}}"
    )
    assert [r.title for r in body_link_records(wikitext, STOP, BUCKETS)] == ["Hà Nội", "Hồ Tây"]


def test_body_link_records_track_sections_and_sentences() -> None:
    wikitext = (
        "Chùa do vua [[Lý Thái Tổ]] cho xây. Câu khác.\n"
        "== Lịch sử hình thành ==\nNăm 1010. Trùng tu thời [[Nhà Trần|Trần]] rất lớn.\n"
        "=== Chi tiết ===\n[[Gạch]] nung.\n"
        "== Du lịch ==\n[[Hà Nội]].\n"
    )
    records = body_link_records(wikitext, STOP, BUCKETS)
    assert [(r.title, r.section) for r in records] == [
        ("Lý Thái Tổ", "lead"), ("Nhà Trần", "history"), ("Gạch", "history"), ("Hà Nội", "other"),
    ]
    assert records[0].before == "Chùa do vua"
    assert records[0].after == "cho xây"
    assert records[1].before == "Trùng tu thời"


def test_unresolved_shared_titles_counts_articles_not_links() -> None:
    links = {
        "P1": [_record("Nhà Lý"), _record("Nhà Lý"), _record("Hà Nội")],
        "P2": [_record("Nhà Lý")],
        "P3": [_record("Lạ"), _record("Lạ")],
    }
    assert unresolved_shared_titles(links, {"Hà Nội": "Q1"}, 2) == {"Nhà Lý"}


def test_follows_any_matches_whole_trailing_word() -> None:
    assert follows_any("nằm trên đường", ("đường", "phố"))
    assert follows_any("góc phố “", ("đường", "phố"))
    assert not follows_any("con đường này do", ("đường", "phố"))


def test_cue_relation_prefers_closest_word_before_then_after() -> None:
    assert cue_relation("xây dựng bởi quan lại thời", "", "person", CUES) == "dated_to_period"
    assert cue_relation("vua", "cho xây năm 1010", "person", CUES) == "built_by"
    assert cue_relation("chùa thờ", "", "deity", CUES) == "worships"
    assert cue_relation("chùa thờ", "", "era", CUES) is None
    assert cue_relation("gần", "và hồ", "person", CUES) is None


def test_collect_mentions_applies_every_filter() -> None:
    links = {
        "P1": [
            _record("Generic"), _record("King", "chùa thờ", section="history"), _record("King", "thời"),
            _record("Street", "đường"), _record("Noun"), _record("Rare"), _record("Self"), _record("Shared"),
        ],
        "P2": [_record("Generic"), _record("Shared"), _record("Place one")],
        "P3": [_record("Generic"), _record("Linked")],
        "P4": [],
    }
    titles = {
        "Generic": "G", "King": "K", "Street": "ST", "Noun": "N", "Rare": "R", "Self": "P1", "Shared": "S",
        "Place one": "P1", "Linked": "L",
    }
    kinds = {"G": "era", "K": "person", "ST": "person", "R": "person", "S": "era", "P1": "place", "L": "place"}
    mentions = collect_mentions(
        links, titles, kinds, graph_nodes={"K", "P1", "L"}, connected={("L", "P3")}, cues=CUES,
        skip_after=("đường",), min_articles=2, max_share=0.5,
    )
    assert [(m.head, m.tail, m.kind, m.section, m.cue) for m in mentions] == [
        ("P1", "K", "person", "history", "worships"),
        ("P1", "S", "era", "lead", None),
        ("P2", "P1", "place", "lead", None),
        ("P2", "S", "era", "lead", None),
    ]
    assert mentions[0].context == "chùa thờ 【King】"


def test_mention_relation_per_scheme() -> None:
    cued = Mention("P1", "K", "person", "history", "worships", "")
    plain = Mention("P1", "S", "era", "lead", None, "")
    assert mention_relation(cued, RelationScheme.GENERIC, "mentions") == "mentions"
    assert mention_relation(cued, RelationScheme.KIND, "mentions") == "mentions_person"
    assert mention_relation(cued, RelationScheme.SECTION, "mentions") == "history_mentions"
    assert mention_relation(cued, RelationScheme.CUE, "mentions") == "worships"
    assert mention_relation(plain, RelationScheme.CUE, "mentions") == "mentions_era"
