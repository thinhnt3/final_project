#!/usr/bin/env python3
"""Image graphs of the full VG ∩ COCO variant over the base vocabulary.

    python -m scripts.build_full_graphs

The base vocabulary (from the 7,000 base train images) is rebuilt, checked against the
base concepts.json and predicates.json, and every full-variant scene graph is mapped onto
it. Validation and test graphs must come out identical to the base ones, so only the
amount of training data differs between the two variants. Writes
<full processed>/graph/{concepts,predicates,image_graphs}.json.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.graph_build import build_image_graph, build_vocabulary
from src.data.json_io import read_json, write_json
from src.data.schemas import SceneGraph, Split

log = logging.getLogger("build_full_graphs")


def main() -> int:
    """Build and check the full-variant graphs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="configs/data/vg_coco.yaml")
    parser.add_argument("--full", default="configs/data/vg_coco_full.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        base, full = DataConfig.from_yaml(args.base), DataConfig.from_yaml(args.full)
        base_dir, full_dir = base.paths.processed_dir, full.paths.processed_dir
        split_of = {int(r["vg_image_id"]): Split(r["split"]) for r in read_json(base_dir / "images.json")}
        base_train = [
            SceneGraph.from_dict(g) for g in read_json(base_dir / "scene_graphs.json")
            if split_of[int(g["vg_image_id"])] is Split.TRAIN
        ]
        vocabulary = build_vocabulary(
            base_train, min_train_images=base.graph.min_train_images, junk_names=base.graph.junk_names
        )
        concepts = [vars(c) for c in vocabulary.concepts]
        predicates = [vars(p) for p in vocabulary.predicates]
        if concepts != read_json(base_dir / "graph" / "concepts.json") or predicates != read_json(
            base_dir / "graph" / "predicates.json"
        ):
            raise DataError("the rebuilt base vocabulary differs from the base concepts/predicates files")

        full_split = {int(r["vg_image_id"]): Split(r["split"]) for r in read_json(full_dir / "images.json")}
        graphs = sorted(
            (SceneGraph.from_dict(g) for g in read_json(full_dir / "scene_graphs.json")), key=lambda g: g.vg_image_id
        )
        image_graphs = [
            build_image_graph(g, full_split[g.vg_image_id], vocabulary, base.graph.junk_names)[0].to_dict()
            for g in graphs
        ]
        base_graphs = {int(g["vg_image_id"]): g for g in read_json(base_dir / "graph" / "image_graphs.json")}
        held_out = [g for g in image_graphs if g["split"] != Split.TRAIN.value]
        differing = [g["vg_image_id"] for g in held_out if base_graphs.get(int(g["vg_image_id"])) != g]
        if differing:
            raise DataError(f"{len(differing)} validation/test graphs differ from the base, e.g. {differing[:5]}")
        out = full_dir / "graph"
        write_json(out / "concepts.json", concepts, indent=None)
        write_json(out / "predicates.json", predicates, indent=None)
        write_json(out / "image_graphs.json", image_graphs, indent=None)
        log.info("%d graphs (%d train) over %d concepts and %d predicates; %d held-out graphs match the base",
                 len(image_graphs), len(image_graphs) - len(held_out), len(concepts), len(predicates), len(held_out))
    except AppError:
        log.exception("building the full graphs failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
