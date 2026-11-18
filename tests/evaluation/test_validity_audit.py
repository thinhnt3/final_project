from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_validity_review import run
from src.core.exceptions import DataError
from src.data.json_io import write_json
from src.data.sources import sha256_of
from src.evaluation.validity_audit import make_audit_manifest, package_audit_manifest


def fixture_cases(tmp_path: Path, count: int = 1):
    photo = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(photo)
    return [
        {
            "case_id": f"13:q{i}",
            "query_id": f"q{i}",
            "text": "A person beside a bicycle.",
            "image_id": 42,
            "image_path": str(photo),
            "image_sha256": sha256_of(photo),
            "predicted_image_is_relevant": True,
            "relations": [
                {
                    "index": 0,
                    "subject": "person",
                    "predicate": "beside",
                    "object": "bicycle",
                    "single_deletion_drop": 0.123,
                    "machine_label": "supported",
                }
            ],
        }
        for i in range(count)
    ]


def manifest_fixture(tmp_path: Path, count: int = 1):
    cases = fixture_cases(tmp_path, count)
    source = tmp_path / "cases.json"
    write_json(source, {"cases": cases})
    manifest = make_audit_manifest(cases, source, "test fixture")
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return manifest, path


def test_package_has_no_fixed_twenty_query_limit_or_machine_answers(tmp_path):
    manifest, path = manifest_fixture(tmp_path, 35)
    manifest["items"][0]["machine_label"] = "secret-machine-answer"
    manifest["items"][0]["single_deletion_drop"] = 0.123
    package = package_audit_manifest(manifest, path)
    assert len(package["items"]) == 35
    assert len(package["photos"]) == 1
    assert package["schema_version"] == "validity-v2-package"
    assert package["photos"]["42"].startswith("data:image/png;base64,")
    text = str(package)
    assert all(
        key not in text for key in ("secret-machine-answer", "single_deletion_drop", "predicted_image_is_relevant")
    )


def test_source_cases_are_untouched_and_item_order_is_seeded(tmp_path):
    cases = fixture_cases(tmp_path, 8)
    original = deepcopy(cases)
    source = tmp_path / "cases.json"
    write_json(source, {"cases": cases})
    assert make_audit_manifest(cases, source, "test") == make_audit_manifest(cases, source, "test")
    assert cases == original


@pytest.mark.parametrize("failure", ["duplicate", "empty", "bool_id", "rubric", "missing_image", "bad_metadata"])
def test_invalid_manifest_is_rejected(tmp_path, failure):
    manifest, path = manifest_fixture(tmp_path)
    if failure == "duplicate":
        manifest["items"].append(manifest["items"][0])
    elif failure == "empty":
        manifest["items"] = []
    elif failure == "bool_id":
        manifest["items"][0]["image_id"] = True
    elif failure == "rubric":
        manifest["rubric_sha256"] = "wrong"
    elif failure == "missing_image":
        manifest["images"].clear()
    else:
        manifest["images"]["42"] = "not image metadata"
    with pytest.raises(DataError):
        package_audit_manifest(manifest, path)


def test_changed_image_is_not_silently_embedded(tmp_path):
    manifest, path = manifest_fixture(tmp_path)
    photo = Path(manifest["images"]["42"]["path"])
    Image.new("RGB", (8, 8), "black").save(photo)
    with pytest.raises(DataError, match="changed"):
        package_audit_manifest(manifest, path)


def test_cli_does_not_overwrite_input_manifest(tmp_path):
    _, path = manifest_fixture(tmp_path)
    original = path.read_bytes()
    with pytest.raises(DataError, match="overwrite"):
        run(Namespace(all=False, per_stratum=None, manifest=str(path), cases=None, output=str(path)))
    assert path.read_bytes() == original


def test_sampling_flags_are_not_silently_ignored_for_existing_manifests(tmp_path):
    with pytest.raises(DataError, match="sampling flags"):
        run(Namespace(all=True, per_stratum=None, manifest="unused.json", cases=None))
