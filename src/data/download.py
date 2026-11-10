from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen


def download_images(manifest: list[dict], out_dir: str | Path) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for item in manifest:
        image_id = int(item["image_id"])
        target = out / f"{image_id}.jpg"
        if not target.exists() or target.stat().st_size == 0:
            request = Request(item["url"], headers={"User-Agent": "project3-graph-retrieval/1.0"})
            with urlopen(request, timeout=60) as response, target.open("wb") as handle:
                handle.write(response.read())
        downloaded.append({**item, "path": str(target), "bytes": target.stat().st_size})
    return downloaded
