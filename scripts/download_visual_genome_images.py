#!/usr/bin/env python3
import json
from pathlib import Path

from src.data.download import download_images


root = Path("data/processed/visual_genome")
manifest = json.loads((root / "images.json").read_text())
downloaded = download_images(manifest, root / "images")
(root / "images_downloaded.json").write_text(json.dumps(downloaded, indent=2))
print(json.dumps({"images": len(downloaded), "bytes": sum(x["bytes"] for x in downloaded)}, indent=2))
