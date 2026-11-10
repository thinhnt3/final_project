#!/usr/bin/env python3
import json
from pathlib import Path

from src.data.splits import make_split, save_split, validate_no_overlap


root = Path("data/processed/visual_genome")
ids = [int(row["image_id"]) for row in json.loads((root / "images.json").read_text())]
split = make_split(ids, seed=13)
validate_no_overlap(split)
save_split(split, "data/splits/visual_genome.json", seed=13)
print(json.dumps(split, indent=2))
