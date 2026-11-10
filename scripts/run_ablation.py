#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.evaluation.ablations import remove_relationships
from src.evaluation.metrics import evaluate
from src.models.embeddings import FrozenCLIPEmbedder
from src.retrieval.index import CosineIndex
from src.retrieval.reranker import rerank


root = Path("data/processed/visual_genome")
images = json.loads((root / "images_downloaded.json").read_text())
graphs = {int(g["image_id"]): g for g in json.loads((root / "graphs.json").read_text())}
queries = json.loads((root / "queries.json").read_text())
split = json.loads(Path("data/splits/visual_genome.json").read_text())
test_ids = set(split["test"])
queries = [q for q in queries if q["relevant_ids"][0] in test_ids]
ids = [int(row["image_id"]) for row in images if int(row["image_id"]) in graphs]
model = FrozenCLIPEmbedder()
vectors = model.encode_images([Image.open(root / "images" / f"{i}.jpg").convert("RGB") for i in ids])
index = CosineIndex(vectors)
outputs = {}
for mode in ("none", "drop_all", "shuffle"):
    rows = []
    mode_graphs = graphs if mode == "none" else {i: remove_relationships(g, mode) for i, g in graphs.items()}
    for query in queries:
        indices, scores = index.search(model.encode_text([query["text"]])[0], len(ids))
        candidate_ids = [ids[int(i)] for i in indices]
        result = rerank(query["text"], candidate_ids, scores, mode_graphs, alpha=0.25)
        rows.append({"ranked_ids": [r["image_id"] for r in result], "relevant_ids": query["relevant_ids"]})
    outputs[mode] = evaluate(rows)
Path("runs/ablation_visual_genome.json").write_text(json.dumps(outputs, indent=2))
print(json.dumps(outputs, indent=2))
