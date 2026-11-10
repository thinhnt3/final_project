#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.evaluation.metrics import evaluate
from src.models.embeddings import FrozenCLIPEmbedder
from src.models.graphsage import GraphSAGEGraphScorer, graph_inputs


root = Path("data/processed/visual_genome")
graphs = {int(g["image_id"]): g for g in json.loads((root / "graphs.json").read_text())}
queries = json.loads((root / "queries.json").read_text())
split = json.loads(Path("data/splits/visual_genome.json").read_text())
train_ids, test_ids = set(split["train"]), set(split["test"])
model = FrozenCLIPEmbedder()
labels = sorted({label for graph in graphs.values() for label in graph["objects"].values()})
label_vectors = {label: torch.from_numpy(model.encode_text([label])[0]).float() for label in labels}
scorer = GraphSAGEGraphScorer()
optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-3)
graph_embeddings = {}
for _epoch in range(8):
    optimizer.zero_grad()
    graph_embeddings = {}
    for image_id, graph in graphs.items():
        x, edge_index = graph_inputs(graph, label_vectors)
        graph_embeddings[image_id] = scorer.encode_graph(x, edge_index)
    loss = torch.tensor(0.0)
    for query in queries:
        positive = query["relevant_ids"][0]
        if positive not in train_ids:
            continue
        negative = next(image_id for image_id in sorted(train_ids) if image_id != positive)
        qvec = torch.from_numpy(model.encode_text([query["text"]])[0]).float()
        loss = loss + torch.relu(0.2 - scorer.score(qvec, graph_embeddings[positive]) + scorer.score(qvec, graph_embeddings[negative]))
    loss.backward()
    optimizer.step()

test_queries = [q for q in queries if q["relevant_ids"][0] in test_ids]
rows = []
for query in test_queries:
    qvec = torch.from_numpy(model.encode_text([query["text"]])[0]).float()
    scores = {image_id: float(scorer.score(qvec, embedding).detach()) for image_id, embedding in graph_embeddings.items()}
    ranked = [i for i, _ in sorted(scores.items(), key=lambda pair: -pair[1])]
    rows.append({"ranked_ids": ranked, "relevant_ids": query["relevant_ids"]})
result = {"test_queries": len(test_queries), "metric": evaluate(rows)}
Path("runs/graphsage_visual_genome.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
