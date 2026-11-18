"""Prepare arbitrary-sized, blinded validity audits without assigning semantic labels."""

from __future__ import annotations

import base64
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.exceptions import DataError
from src.data.sources import sha256_of

RUBRIC = {
    "version": "validity-v2",
    "visual": {
        "supported": "The image visibly supports the subject, directed predicate and object together.",
        "refuted": "Clear visual evidence contradicts the full relation.",
        "uncertain": (
            "Insufficient visibility, ambiguous wording or unresolved object instance. "
            "Absence of clear evidence alone is not refutation."
        ),
    },
    "query": {
        "direct_support": (
            "Supports a central entity/action/constraint explicitly requested; quote an exact query span "
            "and explain. Shared nouns alone are insufficient."
        ),
        "context_only": "Related scene context or incidental attribute, without supporting the requested content.",
        "unrelated": "No support for the query content.",
        "uncertain": "Insufficient clarity to decide relevance.",
    },
}
ITEM_TEXT_FIELDS = ("query_id", "query", "subject", "predicate", "object", "visual_question")


def rubric_sha256(rubric: Mapping[str, Any]) -> str:
    """Hash the exact versioned rubric independently of JSON whitespace."""
    encoded = json.dumps(rubric, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_audit_manifest(
    cases: Sequence[Mapping[str, Any]],
    source_path: Path,
    sampling: str,
    order_seed: int = 20261119,
) -> dict[str, Any]:
    """Convert exported cases to unlabeled audit items; never carry retrieval/AI judgements."""
    items = []
    images = {}
    if not cases:
        raise DataError("an audit requires at least one case")
    for case in cases:
        photo = Path(case["image_path"])
        image_id = case["image_id"]
        if sha256_of(photo) != case["image_sha256"]:
            raise DataError(f"audit image changed: {photo}")
        images[str(image_id)] = {"path": str(photo), "sha256": case["image_sha256"]}
        for relation in case["relations"]:
            # Include the checkpoint case ID so distinct model explanations cannot collide.
            identity = f"{case['case_id']}:{image_id}:{relation['index']}"
            items.append(
                {
                    "audit_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                    "query_id": case["query_id"],
                    "query": case["text"],
                    "image_id": image_id,
                    "relation_index": relation["index"],
                    **{key: relation[key] for key in ("subject", "predicate", "object")},
                    "visual_question": (
                        f"Ảnh có thể hiện quan hệ sau không: {relation['subject']} — "
                        f"{relation['predicate']} — {relation['object']}?"
                    ),
                }
            )

    # Break the source's correct/incorrect grouping before the human sees any item.
    random.Random(order_seed).shuffle(items)
    return {
        "schema_version": "validity-v2",
        "status": "pending_human_audit",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_of(source_path),
        "rubric": RUBRIC,
        "rubric_sha256": rubric_sha256(RUBRIC),
        "order_seed": order_seed,
        "sampling": sampling,
        "images": images,
        "items": items,
    }


def package_audit_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Embed verified images and allowlisted fields for the offline form, with no item limit."""
    if (
        manifest.get("schema_version") != "validity-v2"
        or manifest.get("rubric") != RUBRIC
        or manifest.get("rubric_sha256") != rubric_sha256(RUBRIC)
    ):
        raise DataError("audit manifest must use the supported validity-v2 rubric")
    source_items = manifest.get("items")
    images = manifest.get("images")
    if not isinstance(source_items, list) or not source_items or not isinstance(images, dict):
        raise DataError("audit manifest needs nonempty items and image metadata")

    # Copy only fields that belong in the blinded form; exclude scores and machine labels.
    items = []
    seen = set()
    for item in source_items:
        if not isinstance(item, dict):
            raise DataError("audit item must be a mapping")
        audit_id = item.get("audit_id")
        if not isinstance(audit_id, str) or not audit_id or audit_id in seen:
            raise DataError("audit IDs must be nonempty and unique")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in ITEM_TEXT_FIELDS):
            raise DataError("audit text fields must be nonempty strings")
        if any(type(item.get(key)) is not int or item[key] < 0 for key in ("image_id", "relation_index")):
            raise DataError("audit image and relation IDs must be nonnegative integers")
        if str(item["image_id"]) not in images:
            raise DataError("audit item references missing image metadata")
        items.append({key: item[key] for key in ("audit_id", "image_id", "relation_index", *ITEM_TEXT_FIELDS)})
        seen.add(audit_id)

    # Read each image once, and reject changed/non-image files instead of embedding them.
    photos = {}
    for image_id in {str(item["image_id"]) for item in items}:
        metadata = images[image_id]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or not isinstance(metadata.get("sha256"), str)
        ):
            raise DataError("audit image metadata needs a path and checksum")
        photo = Path(metadata["path"])
        if not photo.is_file() or sha256_of(photo) != metadata["sha256"]:
            raise DataError(f"audit image missing or changed: {photo}")
        try:
            with Image.open(photo) as picture:
                image_format = picture.format
                picture.verify()
        except (OSError, SyntaxError) as exc:
            raise DataError(f"invalid audit image: {photo}") from exc
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(image_format or "")
        if mime is None:
            raise DataError(f"unsupported audit image format: {image_format}")
        photos[image_id] = f"data:{mime};base64,{base64.b64encode(photo.read_bytes()).decode()}"
    return {
        "schema_version": "validity-v2-package",
        "manifest_sha256": sha256_of(manifest_path),
        "rubric_sha256": manifest["rubric_sha256"],
        "items": items,
        "photos": photos,
    }
