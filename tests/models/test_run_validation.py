from dataclasses import asdict, replace

import pytest

from src.core.exceptions import DataError
from src.data.sources import sha256_of
from src.data.yaml_io import write_yaml
from src.models.ablations import Ablation
from src.models.run_validation import validate_run
from src.retrieval.config import RetrievalConfig


def test_run_validation_rejects_changed_input_seed_and_ablation(
    tmp_path,
    monkeypatch,
):
    cfg = RetrievalConfig.from_yaml("configs/retrieval/ablations/full.yaml")
    from types import SimpleNamespace

    from src.data.config import DataConfig

    cfg = replace(cfg, embeddings_dir=tmp_path / "embeddings")
    paths = SimpleNamespace(processed_dir=tmp_path, split_file=tmp_path / "split.json")
    monkeypatch.setattr(
        DataConfig,
        "from_yaml",
        lambda _: SimpleNamespace(paths=paths),
    )
    required = [cfg.model_embeddings_dir() / "meta.json", tmp_path / "graph/image_graphs.json", paths.split_file]
    for file in required:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    source = tmp_path / "source.json"
    source.write_text("{}")
    architecture = asdict(cfg.graphsage)
    architecture.update(ablation="full", image_node="learned")
    saved = {
        "graphsage": architecture,
        "seed": 13,
        "clip_model": cfg.clip.model_name,
        "inputs_sha256": {str(file): sha256_of(file) for file in [source, *required]},
    }
    write_yaml(tmp_path / "config.yaml", saved)
    validate_run(
        tmp_path,
        cfg,
        13,
    )
    with pytest.raises(DataError, match="seed"):
        validate_run(
            tmp_path,
            cfg,
            37,
        )
    with pytest.raises(DataError, match="architecture"):
        validate_run(
            tmp_path,
            replace(cfg, graphsage=replace(cfg.graphsage, ablation=Ablation.OBJECTS_ONLY)),
            13,
        )
    with pytest.raises(DataError, match="requested input"):
        validate_run(
            tmp_path,
            replace(cfg, embeddings_dir=tmp_path / "another"),
            13,
        )
    source.write_text("changed")
    with pytest.raises(DataError, match="changed"):
        validate_run(
            tmp_path,
            cfg,
            13,
        )


def test_resume_does_not_trust_a_lone_result_marker(tmp_path):
    from src.core.exceptions import TrainingError
    from src.models.checkpoint import RunDirectory

    run = RunDirectory(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    saved = {"graphsage": {"ablation": "full"}, "inputs_sha256": {}, "seed": 13}
    write_yaml(tmp_path / "config.yaml", saved)
    with pytest.raises(TrainingError, match="missing"):
        run.validate_finished(saved)
    for f in ("best.pt", "last.pt", "graph_vectors.npy", "graph_vectors.ids.json"):
        (tmp_path / f).touch()
    run.validate_finished(saved)
    with pytest.raises(TrainingError, match="settings changed"):
        run.validate_finished({**saved, "seed": 37})


def test_stale_benchmark_fails_closed(tmp_path):
    from src.models.run_validation import validate_artifact_inputs

    file = tmp_path / "vectors.npy"
    file.write_bytes(b"old")
    report = {"inputs_sha256": {str(file): sha256_of(file)}}
    validate_artifact_inputs(report, "benchmark")
    file.write_bytes(b"new")
    with pytest.raises(DataError, match="stale input"):
        validate_artifact_inputs(report, "benchmark")
    with pytest.raises(DataError, match="missing input provenance"):
        validate_artifact_inputs({}, "benchmark")
