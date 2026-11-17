from pathlib import Path

import pytest

from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.retrieval.config import RetrievalConfig

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "retrieval" / "vg_coco.yaml"


def _raw(**clip_overrides):
    clip = {
        "model_name": "openai/clip-vit-base-patch32", "device": "auto", "image_batch_size": 64, "text_batch_size": 256,
    }
    clip.update(clip_overrides)
    return {
        "data_config": "configs/data/vg_coco.yaml",
        "clip": clip,
        "embeddings_dir": "emb",
        "benchmark": {"recall_ks": [1, 5, 10], "alpha_step": 0.5, "top_k_grid": [10, 50]},
        "runs_dir": "runs",
        "graphsage": {"hidden_dim": 64, "num_layers": 3, "directed": True, "image_node": "learned"},
    }


def test_repo_config_loads_and_points_to_an_existing_data_config():
    config = RetrievalConfig.from_yaml(REPO_CONFIG)
    assert config.clip.device is Device.AUTO
    assert (REPO_CONFIG.parents[2] / config.data_config).is_file()


def test_model_embeddings_dir_is_a_single_directory_per_model():
    config = RetrievalConfig.from_dict(_raw())
    assert config.model_embeddings_dir() == Path("emb/openai--clip-vit-base-patch32")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "gpu"}, "clip.device"),
        ({"model_name": ""}, "model_name"),
        ({"image_batch_size": 0}, "batch_size"),
        ({"text_batch_size": True}, "text_batch_size"),
    ],
)
def test_invalid_clip_settings_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        RetrievalConfig.from_dict(_raw(**overrides))


def test_benchmark_grid_is_parsed():
    benchmark = RetrievalConfig.from_dict(_raw()).benchmark
    assert benchmark.recall_ks == (1, 5, 10)
    assert benchmark.alpha_grid == (0.0, 0.5, 1.0)
    assert benchmark.top_k_grid == (10, 50)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alpha_step": 0.3}, "alpha_step"),
        ({"alpha_step": 0}, "alpha_step"),
        ({"alpha_step": 2}, "alpha_step"),
        ({"alpha_step": "0.1"}, "alpha_step"),
        ({"top_k_grid": []}, "non-empty"),
        ({"top_k_grid": [0]}, "top_k_grid"),
        ({"recall_ks": [1, 1]}, "duplicates"),
        ({"recall_ks": [1.5]}, "recall_ks"),
    ],
)
def test_invalid_benchmark_settings_are_rejected(overrides, message):
    raw = _raw()
    raw["benchmark"].update(overrides)
    with pytest.raises(ConfigError, match=message):
        RetrievalConfig.from_dict(raw)


def test_alpha_grid_runs_from_zero_to_one_in_equal_steps():
    raw = _raw()
    raw["benchmark"]["alpha_step"] = 0.01
    grid = RetrievalConfig.from_dict(raw).benchmark.alpha_grid
    assert len(grid) == 101 and grid[0] == 0.0 and grid[-1] == 1.0 and grid[91] == 0.91
    assert RetrievalConfig.from_yaml(REPO_CONFIG).benchmark.alpha_grid == grid


def test_graphsage_section_is_parsed_and_validated():
    config = RetrievalConfig.from_dict(_raw()).graphsage
    assert (config.hidden_dim, config.num_layers, config.directed, config.image_node.value) == (64, 3, True, "learned")
    for overrides, message in (
        ({"image_node": "zeros"}, "image_node"),
        ({"directed": "yes"}, "directed"),
        ({"num_layers": 0}, "must be >= 1"),
        ({"hidden_dim": 1.5}, "hidden_dim"),
    ):
        raw = _raw()
        raw["graphsage"].update(overrides)
        with pytest.raises(ConfigError, match=message):
            RetrievalConfig.from_dict(raw)
