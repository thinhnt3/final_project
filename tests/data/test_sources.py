import zipfile

import pytest

from src.core.exceptions import ChecksumMismatchError, DataError
from src.data.config import SourceSpec
from src.data.sources import ensure_source, load_manifest, save_manifest, sha256_of


@pytest.fixture
def zipped_source(tmp_path):
    archive = tmp_path / "remote" / "bundle.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/dataset_coco.json", '{"images": []}')
        zf.writestr("nested/other.json", "{}")
    spec = SourceSpec(name="coco", url=archive.as_uri(), target="coco/dataset_coco.json",
                      archive_member="dataset_coco.json")
    return spec, tmp_path / "raw"


def test_fetches_and_extracts_member(zipped_source):
    spec, raw_dir = zipped_source
    entry = ensure_source(spec, raw_dir, {})
    target = raw_dir / spec.target
    assert target.read_text() == '{"images": []}'
    assert entry.sha256 == sha256_of(target)
    assert not any(p.name.startswith("tmp") for p in target.parent.iterdir())


def test_second_run_does_not_refetch(zipped_source, tmp_path):
    spec, raw_dir = zipped_source
    first = ensure_source(spec, raw_dir, {})
    (tmp_path / "remote" / "bundle.zip").unlink()  # refetch would now fail
    assert ensure_source(spec, raw_dir, {"coco": first}) == first


def test_detects_modified_file(zipped_source):
    spec, raw_dir = zipped_source
    entry = ensure_source(spec, raw_dir, {})
    (raw_dir / spec.target).write_text('{"images": [1]}')
    with pytest.raises(ChecksumMismatchError):
        ensure_source(spec, raw_dir, {"coco": entry})


def test_missing_archive_member(zipped_source):
    spec, raw_dir = zipped_source
    bad = SourceSpec(spec.name, spec.url, spec.target, archive_member="absent.json")
    with pytest.raises(DataError, match="absent.json"):
        ensure_source(bad, raw_dir, {})


def test_manifest_roundtrip(zipped_source, tmp_path):
    spec, raw_dir = zipped_source
    entry = ensure_source(spec, raw_dir, {})
    path = tmp_path / "MANIFEST.json"
    save_manifest(path, {"coco": entry})
    assert load_manifest(path) == {"coco": entry}
    assert load_manifest(tmp_path / "none.json") == {}
