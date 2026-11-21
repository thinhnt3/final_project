"""Typed, validated view of `configs/retrieval/mkg_w.yaml`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.kg_graphsage import KgSageConfig
from src.models.train import TrainingConfig
from src.retrieval.config import ClipConfig, _training_config


@dataclass(frozen=True)
class MkgwRetrievalConfig:
    """Root config for retrieval experiments on MKG-W."""

    data_config: Path
    clip: ClipConfig
    embeddings_dir: Path
    recall_ks: tuple[int, ...]
    candidate_ks: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    top_k_grid: tuple[int, ...]
    path_decay_grid: tuple[float, ...]
    path_shuffle_seed: int
    graphsage: KgSageConfig
    training: TrainingConfig
    runs_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> MkgwRetrievalConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        return cls.from_dict(read_mapping(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MkgwRetrievalConfig:
        """Build from an already-parsed mapping (used by tests)."""
        clip_raw = require_section(raw, "clip")
        device = require_field(clip_raw, "device", str)
        if device not in {d.value for d in Device}:
            raise ConfigError(f"clip.device must be one of {[d.value for d in Device]}, got {device!r}")
        clip = ClipConfig(
            model_name=require_field(clip_raw, "model_name", str),
            device=Device(device),
            image_batch_size=require_field(clip_raw, "image_batch_size", int),
            text_batch_size=require_field(clip_raw, "text_batch_size", int),
        )
        if not clip.model_name or clip.image_batch_size < 1 or clip.text_batch_size < 1:
            raise ConfigError("clip.model_name must be set and batch sizes must be >= 1")
        benchmark_raw = require_section(raw, "benchmark")
        alpha_step = float(require_field(benchmark_raw, "alpha_step", (int, float)))
        steps = round(1.0 / alpha_step) if alpha_step > 0 else 0
        if not 0.0 < alpha_step <= 1.0 or abs(steps * alpha_step - 1.0) > 1e-9:
            raise ConfigError(f"benchmark.alpha_step must divide 1 evenly and lie in (0, 1], got {alpha_step}")
        path_raw = require_section(raw, "path_score")
        decays = path_raw.get("decay_grid")
        if (not isinstance(decays, list) or not decays
                or not all(isinstance(d, (int, float)) and not isinstance(d, bool) and 0 < d <= 1 for d in decays)):
            raise ConfigError("path_score.decay_grid must be a non-empty list of numbers in (0, 1]")
        return cls(
            data_config=Path(require_field(raw, "data_config", str)),
            clip=clip,
            embeddings_dir=Path(require_field(raw, "embeddings_dir", str)),
            recall_ks=_positive_ints(benchmark_raw, "recall_ks"),
            candidate_ks=_positive_ints(benchmark_raw, "candidate_ks"),
            alpha_grid=tuple(round(i / steps, 6) for i in range(steps + 1)),
            top_k_grid=_positive_ints(benchmark_raw, "top_k_grid"),
            path_decay_grid=tuple(float(d) for d in decays),
            path_shuffle_seed=require_field(path_raw, "shuffle_seed", int),
            graphsage=KgSageConfig(
                hidden_dim=require_field(require_section(raw, "kg_graphsage"), "hidden_dim", int),
                num_layers=require_field(require_section(raw, "kg_graphsage"), "num_layers", int),
            ),
            training=_training_config(raw),
            runs_dir=Path(require_field(raw, "runs_dir", str)),
        )

    def model_embeddings_dir(self) -> Path:
        """Directory holding the embedding tables of the configured CLIP model."""
        return self.embeddings_dir / self.clip.model_name.replace("/", "--")


def _positive_ints(raw: dict[str, Any], key: str) -> tuple[int, ...]:
    values = raw.get(key)
    if (not isinstance(values, list) or not values
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in values)):
        raise ConfigError(f"benchmark.{key} must be a non-empty list of integers >= 1")
    return tuple(values)
