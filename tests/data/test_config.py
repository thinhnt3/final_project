from pathlib import Path

import pytest

from src.core.exceptions import ConfigError
from src.data.config import DataConfig

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "vg_coco.yaml"


SPLIT = {
    "seed": 13,
    "sizes": {"train": 3500, "val": 750, "test": 750},
    "min_relations": 1,
    "n_strata": 4,
    "exclude_ids": [],
}


def _raw(**overrides):
    raw = {
        "paths": {
            "raw_dir": "r", "interim_dir": "i", "manifest_file": "r/M.json",
            "split_file": "s.json", "processed_dir": "p", "graph_card_file": "g.yaml",
        },
        "sources": [{"name": "a", "url": "file:///a", "target": "a.json"}],
        "join": {"captions_per_image": 5},
        "normalize": {"backfill_min_count": 5, "backfill_min_share": 0.9},
        "split": dict(SPLIT),
        "download": {"workers": 8, "retries": 3, "timeout_s": 60},
        "graph": {"min_train_images": 5, "junk_names": ["the", "white"]},
    }
    raw.update(overrides)
    return raw


def test_repo_config_loads():
    config = DataConfig.from_yaml(REPO_CONFIG)
    assert config.join.captions_per_image == 5
    assert config.raw_path("karpathy_coco") == Path("data/raw/coco/dataset_coco.json")


def test_rejects_duplicate_source_names():
    source = {"name": "a", "url": "u", "target": "t"}
    with pytest.raises(ConfigError, match="duplicate"):
        DataConfig.from_dict(_raw(sources=[source, source]))


@pytest.mark.parametrize("value", [0, "5", True])
def test_rejects_bad_captions_per_image(value):
    with pytest.raises(ConfigError):
        DataConfig.from_dict(_raw(join={"captions_per_image": value}))


@pytest.mark.parametrize("normalize", [
    {"backfill_min_count": 0, "backfill_min_share": 0.9},
    {"backfill_min_count": 5, "backfill_min_share": 0.5},
    {"backfill_min_count": 5, "backfill_min_share": 1.2},
    {"backfill_min_count": 5},
])
def test_rejects_bad_normalize(normalize):
    with pytest.raises(ConfigError):
        DataConfig.from_dict(_raw(normalize=normalize))


@pytest.mark.parametrize("override", [
    {"sizes": {"train": 3500, "val": 750}},
    {"sizes": {"train": 3500, "val": 750, "test": 0}},
    {"exclude_ids": ["12"]},
    {"n_strata": 0},
])
def test_rejects_bad_split(override):
    with pytest.raises(ConfigError):
        DataConfig.from_dict(_raw(split={**SPLIT, **override}))


def test_split_exclude_ids_sorted_and_deduplicated():
    config = DataConfig.from_dict(_raw(split={**SPLIT, "exclude_ids": [9, 3, 9]}))
    assert config.split.exclude_ids == (3, 9)


@pytest.mark.parametrize("download", [
    {"workers": 0, "retries": 3, "timeout_s": 60},
    {"workers": 8, "retries": 0, "timeout_s": 60},
    {"workers": 8, "retries": 3, "timeout_s": 0},
])
def test_rejects_bad_download(download):
    with pytest.raises(ConfigError):
        DataConfig.from_dict(_raw(download=download))


def test_unknown_source_raises():
    with pytest.raises(ConfigError, match="unknown source"):
        DataConfig.from_dict(_raw()).source("missing")


def test_graph_section_is_validated():
    config = DataConfig.from_dict(_raw())
    assert config.graph.min_train_images == 5
    assert config.graph.junk_names == frozenset({"the", "white"})
    with pytest.raises(ConfigError, match="min_train_images"):
        DataConfig.from_dict(_raw(graph={"min_train_images": 0, "junk_names": []}))
    with pytest.raises(ConfigError, match="lowercase"):
        DataConfig.from_dict(_raw(graph={"min_train_images": 5, "junk_names": ["The "]}))
    with pytest.raises(ConfigError, match="duplicates"):
        DataConfig.from_dict(_raw(graph={"min_train_images": 5, "junk_names": ["a", "a"]}))
