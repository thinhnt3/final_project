from __future__ import annotations

from typing import Any

import numpy as np

from src.models.embeddings import cosine_search
from src.retrieval.reranker import rerank


def run_query(query: str, relevant_ids: list[int], image_ids: list[int], image_vectors: np.ndarray,
              graphs: dict[int, dict[str, Any]], text_encoder: Any, alpha: float = 0.7) -> tuple[dict, dict]:
    query_vector = text_encoder.encode_text([query])[0]
    indices, scores = cosine_search(query_vector, image_vectors, len(image_ids))
    candidate_ids = [image_ids[i] for i in indices]
    clip_row = {"ranked_ids": candidate_ids, "relevant_ids": relevant_ids}
    graph_results = rerank(query, candidate_ids, scores, graphs, alpha=alpha)
    graph_row = {"ranked_ids": [row["image_id"] for row in graph_results], "relevant_ids": relevant_ids,
                 "results": graph_results}
    return clip_row, graph_row
