"""Small, dependency-light loader for the official Visual Genome dumps."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def normalize_scene_graphs(records: list[dict[str, Any]], image_ids: set[int] | None = None) -> dict[int, dict[str, Any]]:
    """Return compact graph records keyed by image id."""
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        image_id = int(record["image_id"])
        if image_ids is not None and image_id not in image_ids:
            continue
        objects = {
            int(obj["object_id"]): str((obj.get("name") or (obj.get("names") or [""])[0])).lower().strip()
            for obj in record.get("objects", [])
            if obj.get("name") or obj.get("names")
        }
        relationships = []
        for rel in record.get("relationships", []):
            subject_id = int(rel.get("subject_id", -1))
            object_id = int(rel.get("object_id", -1))
            if subject_id not in objects or object_id not in objects or not rel.get("predicate"):
                continue
            relationships.append({
                "subject_id": subject_id,
                "object_id": object_id,
                "subject": objects[subject_id],
                "predicate": str(rel["predicate"]).lower().strip(),
                "object": objects[object_id],
            })
        if objects and relationships:
            result[image_id] = {"image_id": image_id, "objects": objects, "relationships": relationships}
    return result


def graph_text(graph: dict[str, Any]) -> str:
    parts = list(graph["objects"].values())
    parts.extend(f"{r['subject']} {r['predicate']} {r['object']}" for r in graph["relationships"])
    return ". ".join(parts)


def graph_vocabulary(graph: dict[str, Any]) -> set[str]:
    tokens = set(graph["objects"].values())
    for rel in graph["relationships"]:
        tokens.update((rel["subject"], rel["predicate"], rel["object"]))
    return tokens
