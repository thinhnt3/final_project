#!/usr/bin/env python3
"""Legacy: download the 40-image subset. Remove when Phase 1 moves to vg_coco."""
import json
from pathlib import Path

from src.data.download import ImageJob, download_images, image_path

root = Path("data/processed/visual_genome")
manifest = json.loads((root / "images.json").read_text())
out_dir = root / "images"
report = download_images(
    [ImageJob(int(item["image_id"]), item["url"]) for item in manifest],
    out_dir, workers=4, retries=3, timeout_s=60,
)
if report.failed:
    raise SystemExit(f"failed downloads: {report.failed}")
downloaded = []
for item in manifest:
    path = image_path(out_dir, int(item["image_id"]))
    downloaded.append({**item, "path": str(path), "bytes": path.stat().st_size})
(root / "images_downloaded.json").write_text(json.dumps(downloaded, indent=2))
print(json.dumps({"images": len(downloaded), "bytes": sum(x["bytes"] for x in downloaded)}, indent=2))
