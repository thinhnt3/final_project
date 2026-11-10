#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate
from src.models.embeddings import HashEmbedder, cosine_search
from src.retrieval.pipeline import run_query


def main() -> None:
    root = Path("data/processed/visual_genome")
    graphs = {int(g["image_id"]): g for g in json.loads((root / "graphs.json").read_text())}
    queries = json.loads((root / "queries.json").read_text())
    ids = sorted(graphs)
    embedder = HashEmbedder()
    # The fallback tests pipeline contracts without downloading a model or image archive.
    image_vectors = embedder.encode_text([" ".join(graphs[i]["objects"].values()) for i in ids])
    rows_clip, rows_graph = [], []
    for q in queries:
        clip_row, graph_row = run_query(q["text"], q["relevant_ids"], ids, image_vectors, graphs, embedder, alpha=0.7)
        rows_clip.append(clip_row)
        rows_graph.append(graph_row)
    print(json.dumps({"clip_only": evaluate(rows_clip), "clip_graph": evaluate(rows_graph)}, indent=2))


if __name__ == "__main__":
    main()
