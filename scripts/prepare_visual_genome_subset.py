#!/usr/bin/env python3
"""Create a small reproducible VG subset and graph-derived text queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.visual_genome import load_json, normalize_scene_graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="data/raw/visual_genome/scene_graphs.json")
    parser.add_argument("--image-data", default="data/raw/visual_genome/image_data.json")
    parser.add_argument("--out", default="data/processed/visual_genome")
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = normalize_scene_graphs(load_json(args.graphs))
    metadata = {int(x["image_id"]): x for x in load_json(args.image_data)}
    selected = [g for image_id, g in sorted(graphs.items()) if image_id in metadata and len(g["relationships"]) >= 2][:args.images]
    ids = {g["image_id"] for g in selected}
    manifest = [{"image_id": i, "url": metadata[i].get("url"), "width": metadata[i].get("width"), "height": metadata[i].get("height")} for i in sorted(ids)]
    queries = []
    for g in selected:
        rel = g["relationships"][0]
        queries.append({"query_id": f"vg-{g['image_id']}-0", "text": f"a {rel['subject']} {rel['predicate']} {rel['object']}", "relevant_ids": [g["image_id"]], "type": "indirect"})
    (out / "images.json").write_text(json.dumps(manifest, indent=2))
    (out / "graphs.json").write_text(json.dumps(selected, indent=2))
    (out / "queries.json").write_text(json.dumps(queries, indent=2))
    print(json.dumps({"images": len(manifest), "queries": len(queries), "graphs": sum(len(g["relationships"]) for g in selected)}, indent=2))


if __name__ == "__main__":
    main()
