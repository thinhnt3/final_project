#!/usr/bin/env python3
"""Split for the full VG ∩ COCO variant: the base split's val and test, every other eligible image in train.

    python -m scripts.make_full_split --base data/splits/vg_coco.json --config configs/data/vg_coco_full.yaml

Eligible means the base build's rules (`select_candidates`: a scene graph with at least
min_relations relations, not excluded). Karpathy-test images stay out of train, as in the
base split, so no test-pool image is ever trained on.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.core.exceptions import AppError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import KarpathySplit, Split
from src.data.sources import sha256_of
from src.data.splits import select_candidates, validate_no_overlap

log = logging.getLogger("make_full_split")


def main() -> int:
    """Write the full split file named in the config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="split file whose val and test are kept")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = DataConfig.from_yaml(args.config)
        interim = config.paths.interim_dir
        images_path, graphs_path = interim / "images.json", interim / "scene_graphs.json"
        karpathy = {int(r["vg_image_id"]): KarpathySplit(r["karpathy_split"]) for r in read_json(images_path)}
        relations = {int(g["vg_image_id"]): len(g["relations"]) for g in read_json(graphs_path)}
        candidates = select_candidates(
            karpathy, relations, min_relations=config.split.min_relations, exclude_ids=config.split.exclude_ids
        )
        base = read_json(args.base)
        held_out = set(base[Split.VAL.value]) | set(base[Split.TEST.value])
        train = sorted(
            c.vg_image_id for c in candidates
            if c.karpathy_split is not KarpathySplit.TEST and c.vg_image_id not in held_out
        )
        split = {
            Split.TRAIN.value: train,
            Split.VAL.value: base[Split.VAL.value],
            Split.TEST.value: base[Split.TEST.value],
        }
        validate_no_overlap(split)
        write_json(config.paths.split_file, {
            "config": {"base_split": args.base, **config.split.to_dict()},
            "inputs_sha256": {str(p): sha256_of(p) for p in (images_path, graphs_path)},
            "candidates": len(candidates),
            **split,
        }, indent=None)
        log.info("train=%d val=%d test=%d → %s", len(train), len(split["val"]), len(split["test"]),
                 config.paths.split_file)
    except AppError:
        log.exception("full split failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
