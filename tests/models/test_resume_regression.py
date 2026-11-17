"""Regression for the metadata-only completion check, reproduced explicitly in history."""
import pytest

from src.core.exceptions import TrainingError
from src.data.yaml_io import write_yaml
from src.models.checkpoint import RunDirectory


def test_finished_run_requires_local_artifacts(tmp_path):
    run = RunDirectory(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    settings = {"graphsage": {}, "seed": 13, "inputs_sha256": {}}
    write_yaml(tmp_path / "config.yaml", settings)

    # A completed-result marker must not bypass missing checkpoint/vector checks.
    with pytest.raises(TrainingError, match="missing"):
        run.validate_finished(settings)
