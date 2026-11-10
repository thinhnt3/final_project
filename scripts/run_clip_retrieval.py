#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from src.evaluation.metrics import evaluate
from src.models.embeddings import FrozenCLIPEmbedder
from src.retrieval.index import CosineIndex
from src.retrieval.reranker import rerank


root = Path("data/processed/visual_genome")
images = json.loads((root / "images_downloaded.json").read_text())
graphs = {int(g["image_id"]): g for g in json.loads((root / "graphs.json").read_text())}
queries = json.loads((root / "queries.json").read_text())
split = json.loads(Path("data/splits/visual_genome.json").read_text())
ids = [int(row["image_id"]) for row in images if int(row["image_id"]) in graphs]
paths = [root / "images" / f"{image_id}.jpg" for image_id in ids]
model = FrozenCLIPEmbedder()
image_vectors = model.encode_images([Image.open(path).convert("RGB") for path in paths])
index = CosineIndex(image_vectors)
def run(alpha, selected_queries):
    clip_rows, graph_rows = [], []
    for query in selected_queries:
        indices, scores = index.search(model.encode_text([query["text"]])[0], len(ids))
        candidate_ids = [ids[int(i)] for i in indices]
        clip_rows.append({"ranked_ids": candidate_ids, "relevant_ids": query["relevant_ids"]})
        results = rerank(query["text"], candidate_ids, scores, graphs, alpha=alpha)
        graph_rows.append({"ranked_ids": [r["image_id"] for r in results], "relevant_ids": query["relevant_ids"], "results": results})
    return evaluate(clip_rows), evaluate(graph_rows)

val_ids = set(split["val"])
test_ids = set(split["test"])
val_queries = [q for q in queries if q["relevant_ids"][0] in val_ids]
test_queries = [q for q in queries if q["relevant_ids"][0] in test_ids]
alpha = max((0.0, 0.25, 0.5, 0.7, 1.0), key=lambda a: run(a, val_queries)[1]["mrr"])
seed_results = []
for seed in (13, 37, 97):
    random.seed(seed)
    seed_results.append({"seed": seed, "clip_only": run(1.0, test_queries)[0], "clip_graph": run(alpha, test_queries)[1]})

def aggregate(key, metric):
    values = [row[key][metric] for row in seed_results]
    import statistics
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values)}

output = {"backend": "openai/clip-vit-base-patch32", "images": len(ids), "val_queries": len(val_queries),
          "test_queries": len(test_queries), "alpha": alpha, "seeds": seed_results,
          "aggregate": {key: {metric: aggregate(key, metric) for metric in ("recall@1", "recall@5", "recall@10", "mrr")} for key in ("clip_only", "clip_graph")}}
Path("runs").mkdir(exist_ok=True)
(Path("runs") / "clip_visual_genome.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
