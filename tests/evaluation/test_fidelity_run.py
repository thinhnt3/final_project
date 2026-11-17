from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts.evaluate_fidelity import run
from src.core.exceptions import ConfigError, DataError
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.embeddings.store import EmbeddingTable, save_table
from src.models.checkpoint import SETTINGS_FILE, RunDirectory
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature
from src.retrieval.config import RetrievalConfig


@pytest.fixture
def artifacts(tmp_path):
    processed = tmp_path / "data"
    graph_path, query_path = processed / "graph/image_graphs.json", processed / "queries.json"
    split_path = tmp_path / "split.json"
    write_json(graph_path, [{"vg_image_id": 7, "concept_ids": [0, 1, 2], "triplets": [[0, 0, 1], [0, 1, 2]]}])
    write_json(query_path, [{"query_id": "q-test", "split": "test", "relevant_ids": [7]}])
    write_json(split_path, {"test": [7]})
    write_json(processed / "images.json", [{"vg_image_id": 7, "split": "test"}])
    data_raw = yaml.safe_load(Path("configs/data/vg_coco.yaml").read_text())
    data_raw["paths"].update(processed_dir=str(processed), split_file=str(split_path))
    data_path = tmp_path / "data.yaml"
    data_path.write_text(yaml.safe_dump(data_raw))
    config = replace(
        RetrievalConfig.from_yaml("configs/retrieval/vg_coco.yaml"),
        data_config=data_path,
        embeddings_dir=tmp_path / "embeddings",
        runs_dir=tmp_path / "runs",
    )
    embeddings = config.model_embeddings_dir()
    torch.manual_seed(13)
    unit = lambda rows: torch.nn.functional.normalize(torch.randn(rows, 8), dim=-1).numpy()  # noqa: E731
    values = {
        "images": ((7,), unit(1)),
        "queries": (("q-test",), unit(1)),
        "concepts": ((0, 1, 2), unit(3)),
        "predicates": ((0, 1), unit(2)),
    }
    for name, (ids, vectors) in values.items():
        save_table(embeddings, name, EmbeddingTable(ids, vectors))
    meta_path = embeddings / "meta.json"
    write_json(meta_path, {"inputs_sha256": {str(query_path): sha256_of(query_path)}, "tables": {}})
    hashes = {str(p): sha256_of(p) for p in (graph_path, split_path, meta_path)}
    model_config = GraphSageConfig(8, 3, True, ImageNodeFeature.LEARNED)
    model = GraphImageEncoder(
        model_config, torch.from_numpy(values["concepts"][1]), torch.from_numpy(values["predicates"][1])
    )
    directory = RunDirectory(config.runs_dir / "graphsage/saved-run/seed13")
    directory.prepare(resume=False, overwrite=False)
    directory.save_best({"model": model.state_dict()})
    settings = {
        "clip_model": config.clip.model_name,
        "seed": 13,
        "graphsage": {**asdict(model_config), "image_node": "learned"},
        "inputs_sha256": hashes,
    }
    (directory.path / SETTINGS_FILE).write_text(yaml.safe_dump(settings))
    benchmark_path = config.runs_dir / "benchmark/graphsage-saved-run.json"
    write_json(
        benchmark_path,
        {"inputs_sha256": hashes, "seeds": {"13": {"selected_on_validation": {"alpha": 0.8, "top_k": 1}}}},
    )
    return config, graph_path, benchmark_path


def test_uses_saved_architecture_and_validation_alpha_without_training(artifacts):
    config, _, _ = artifacts
    first = run(config, "saved-run", 13, Split.TEST, 1, 3)
    assert first == run(config, "saved-run", 13, Split.TEST, 1, 3)
    assert first["alpha"] == 0.8 and first["top_k_selected_on_validation"] == 1
    assert first["queries"][0]["query_id"] == "q-test"
    assert first["queries"][0]["image_id"] == 7
    assert first["inputs_sha256"] and first["budget"] == 3
    # Current config defaults to hidden_dim=512; checkpoint is hidden_dim=8.
    assert config.graphsage.hidden_dim == 512


def test_stale_graph_and_missing_validation_parameters_are_rejected(artifacts):
    config, graph_path, benchmark_path = artifacts
    benchmark = read_json(benchmark_path)
    benchmark["seeds"] = {}
    write_json(benchmark_path, benchmark)
    with pytest.raises(DataError, match="validation-selected"):
        run(config, "saved-run", 13, Split.TEST, 1, 3)
    graph_path.write_text("[]")
    with pytest.raises(DataError, match="inputs changed"):
        run(config, "saved-run", 13, Split.TEST, 1, 3)


def test_invalid_sample_and_split_do_not_trigger_evaluation(artifacts):
    config, _, _ = artifacts
    with pytest.raises(ConfigError, match="exceeds"):
        run(config, "saved-run", 13, Split.TEST, 2, 3)
    with pytest.raises(ConfigError):
        run(config, "saved-run", 13, Split.TRAIN, 1, 3)


def test_cli_refuses_to_overwrite_results(tmp_path, monkeypatch):
    from scripts.evaluate_fidelity import main

    output = tmp_path / "result.json"
    output.write_text("original")
    monkeypatch.setattr("sys.argv", ["evaluate_fidelity", "--output", str(output)])
    assert main() == 1
    assert output.read_text() == "original"
