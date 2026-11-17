"""Verify that exported graph vectors/checkpoints belong to the requested experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping
from src.core.exceptions import DataError
from src.data.config import DataConfig
from src.data.sources import sha256_of
from src.embeddings.store import META_FILE
from src.models.checkpoint import SETTINGS_FILE
from src.retrieval.config import RetrievalConfig


def validate_run(
    path: Path,
    config: RetrievalConfig,
    seed: int,
) -> None:
    """Reject mismatched architecture, seed, CLIP model, or changed training inputs.

    Legacy snapshots without ablation fields mean full/0. Snapshot paths are the
    repository-relative paths actually used for training; absent inputs fail closed.
    """
    # Match the requested architecture and seed to the training snapshot.
    saved = read_mapping(path / SETTINGS_FILE)
    architecture = {"ablation": "full", "ablation_seed": 0, **saved.get("graphsage", {})}
    if architecture != asdict(config.graphsage):
        raise DataError(f"{path}: architecture/ablation does not match the requested config")
    if saved.get("seed") != seed or saved.get("clip_model") != config.clip.model_name:
        raise DataError(f"{path}: seed or CLIP model does not match")

    # Reject inputs that changed after training.
    hashes = saved.get("inputs_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise DataError(f"{path}: no input provenance in the training snapshot")
    for name, expected in hashes.items():
        source = Path(name)
        if not source.is_file() or sha256_of(source) != expected:
            raise DataError(f"{path}: training input missing or changed: {source}")

    # Prevent a valid old run from being scored against another dataset/cache.
    data = DataConfig.from_yaml(config.data_config)
    required = (
        config.model_embeddings_dir() / META_FILE,
        data.paths.processed_dir / "graph/image_graphs.json",
        data.paths.split_file,
    )
    resolved = {Path(name).resolve(): value for name, value in hashes.items()}
    for source in required:
        if source.resolve() not in resolved:
            raise DataError(f"{path}: requested input is not part of this training run: {source}")


def validate_artifact_inputs(
    report: dict[str, Any],
    label: str,
) -> None:
    """Reject a stale benchmark before consuming its validation-selected alpha/k."""
    # A benchmark selection is reusable only with its recorded inputs.
    hashes = report.get("inputs_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise DataError(f"{label}: missing input provenance")
    for name, expected in hashes.items():
        source = Path(name)
        if not source.is_file() or sha256_of(source) != expected:
            raise DataError(f"{label}: stale input: {source}")
