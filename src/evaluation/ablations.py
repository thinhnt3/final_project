from __future__ import annotations

from copy import deepcopy


def remove_relationships(graph: dict, mode: str = "none") -> dict:
    result = deepcopy(graph)
    if mode == "drop_all":
        result["relationships"] = []
    elif mode == "shuffle":
        result["relationships"] = [dict(r, predicate="random_relation") for r in result["relationships"]]
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    return result
