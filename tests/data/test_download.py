import pytest
from PIL import Image

from src.data.download import ImageJob, download_images, image_path, is_valid_image


@pytest.fixture
def remote(tmp_path):
    """A 'remote' folder with one good JPEG and one corrupt file, served via file://."""
    folder = tmp_path / "remote"
    folder.mkdir()
    Image.new("RGB", (8, 8), "red").save(folder / "good.jpg", format="JPEG")
    (folder / "bad.jpg").write_bytes(b"not an image")
    return folder


def _run(jobs, out_dir):
    return download_images(jobs, out_dir, workers=2, retries=1, timeout_s=5)


def test_downloads_valid_and_reports_failures(remote, tmp_path):
    out = tmp_path / "out"
    jobs = [
        ImageJob(1, (remote / "good.jpg").as_uri()),
        ImageJob(2, (remote / "bad.jpg").as_uri()),
        ImageJob(3, (remote / "missing.jpg").as_uri()),
    ]
    report = _run(jobs, out)
    assert report.downloaded == 1
    assert sorted(f["image_id"] for f in report.failed) == [2, 3]
    assert is_valid_image(image_path(out, 1))
    assert not image_path(out, 2).exists()
    assert not list(out.glob("*.part"))


def test_skips_images_already_present(remote, tmp_path):
    out = tmp_path / "out"
    job = ImageJob(1, (remote / "good.jpg").as_uri())
    _run([job], out)
    (remote / "good.jpg").unlink()  # a refetch would now fail
    report = _run([job], out)
    assert (report.already_present, report.downloaded, report.failed) == (1, 0, [])


def test_corrupt_local_file_is_refetched(remote, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    image_path(out, 1).write_bytes(b"truncated")
    report = _run([ImageJob(1, (remote / "good.jpg").as_uri())], out)
    assert report.downloaded == 1
    assert is_valid_image(image_path(out, 1))
