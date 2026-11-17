"""Graph score without a GNN: IDF-weighted overlap between caption words and graph words.

Baseline for the learned graph score. It answers how far the graph alone can rank images
when matching is done on words. No file I/O here.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from src.core.exceptions import DataError
from src.data.review import content_tokens
from src.data.schemas import Split
from src.evaluation.benchmark import SplitData

_WORD = re.compile(r"[a-z]+")
# Words this short are left alone when stripping a plural `s` ("bus", "gas").
_MIN_PLURAL_LENGTH = 4


def stem(word: str) -> str:
    """Drop one plural `s` so `bikes` matches `bike`; `glass` and short words are kept.

    Deliberately crude: `glasses` and `men` do not reach `glass` and `man`.
    """
    if len(word) >= _MIN_PLURAL_LENGTH and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def text_stems(text: str) -> set[str]:
    """Stems of the content words of a caption (function words removed)."""
    return {stem(token) for token in content_tokens(text)}


def graph_stems(
    image_graph: Mapping[str, Any],
    concept_text: Mapping[int, str],
    predicate_text: Mapping[int, str],
    *,
    use_predicates: bool,
) -> set[str]:
    """Stems of all words in an image graph: concept texts, plus predicate texts if asked."""
    texts = [concept_text[c] for c in image_graph["concept_ids"]]
    if use_predicates:
        texts += [predicate_text[p] for _, p, _ in image_graph["triplets"]]
    return {stem(word) for text in texts for word in _WORD.findall(text.lower())}


def inverse_document_frequency(train_stem_sets: Iterable[set[str]]) -> tuple[dict[str, float], float]:
    """IDF of each stem over train image graphs, and the IDF of a stem never seen in train.

    idf = ln(1 + N / (1 + df)), so a stem in every image still weighs a little and an
    unseen stem weighs the most.
    """
    counts: Counter[str] = Counter()
    n_images = 0
    for stems in train_stem_sets:
        counts.update(stems)
        n_images += 1
    if n_images == 0:
        raise DataError("cannot compute IDF from zero train images")
    idf = {word: math.log(1.0 + n_images / (1.0 + df)) for word, df in counts.items()}
    return idf, math.log(1.0 + n_images)


def lexical_scores(
    query_texts: Sequence[str],
    gallery_stem_sets: Sequence[set[str]],
    idf: Mapping[str, float],
    unseen_idf: float,
) -> np.ndarray:
    """(queries × gallery) score in [0, 1]: IDF weight of the caption stems found in the image
    graph, over the IDF weight of all caption stems. A caption with no content words scores 0.
    """
    if not query_texts or not gallery_stem_sets:
        raise DataError("lexical scoring needs at least one query and one gallery image")
    column_of = {word: column for column, word in enumerate(sorted(set().union(*gallery_stem_sets)))}
    contains = np.zeros((len(gallery_stem_sets), len(column_of)), dtype=np.float64)
    for row, stems in enumerate(gallery_stem_sets):
        contains[row, [column_of[word] for word in stems]] = 1.0

    weights = np.zeros((len(query_texts), len(column_of)), dtype=np.float64)
    totals = np.zeros(len(query_texts), dtype=np.float64)
    for row, text in enumerate(query_texts):
        for word in text_stems(text):
            weight = idf.get(word, unseen_idf)
            totals[row] += weight
            if word in column_of:
                weights[row, column_of[word]] = weight
    matched = weights @ contains.T
    return np.divide(matched, totals[:, None], out=np.zeros_like(matched), where=totals[:, None] > 0)


def lexical_graph_scores(
    image_graphs: Sequence[Mapping[str, Any]],
    concept_text: Mapping[int, str],
    predicate_text: Mapping[int, str],
    query_text: Mapping[str, str],
    data: Mapping[Split, SplitData],
    *,
    use_predicates: bool,
) -> dict[Split, np.ndarray]:
    """Lexical graph score matrix of each split, rows and columns in the split's id order.

    IDF is computed from the graphs of train images only.
    """
    stems = {
        int(graph["vg_image_id"]): graph_stems(graph, concept_text, predicate_text, use_predicates=use_predicates)
        for graph in image_graphs
    }
    idf, unseen_idf = inverse_document_frequency(
        stems[int(graph["vg_image_id"])] for graph in image_graphs if graph["split"] == Split.TRAIN.value
    )
    scores = {}
    for split, split_data in data.items():
        without_graph = [image_id for image_id in split_data.image_ids if image_id not in stems]
        without_text = [query_id for query_id in split_data.query_ids if query_id not in query_text]
        if without_graph or without_text:
            raise DataError(
                f"split '{split.value}': {len(without_graph)} images without a graph, "
                f"{len(without_text)} queries without text, e.g. {(without_graph + without_text)[:5]}"
            )
        scores[split] = lexical_scores(
            [query_text[query_id] for query_id in split_data.query_ids],
            [stems[image_id] for image_id in split_data.image_ids],
            idf,
            unseen_idf,
        )
    return scores
