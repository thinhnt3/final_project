"""Typed, validated view of `configs/retrieval/vg_coco.yaml`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.graphsage import GraphSageConfig, ImageNodeFeature

_ALPHA_STEP_TOLERANCE = 1e-9
_ALPHA_DECIMALS = 6


@dataclass(frozen=True)
class ClipConfig:
    """Frozen CLIP model and batch sizes for encoding."""

    model_name: str
    device: Device
    image_batch_size: int
    text_batch_size: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Metric cut-offs and the validation search grid for reranking.

    `alpha_grid` is derived from `alpha_step`: 0, step, ..., 1.
    """

    recall_ks: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    top_k_grid: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalConfig:
    """Root config for retrieval experiments."""

    data_config: Path
    clip: ClipConfig
    embeddings_dir: Path
    benchmark: BenchmarkConfig
    runs_dir: Path
    graphsage: GraphSageConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> RetrievalConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        return cls.from_dict(read_mapping(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RetrievalConfig:
        """Build from an already-parsed mapping (used by tests)."""
        clip_raw = require_section(raw, "clip")
        device_raw = require_field(clip_raw, "device", str)
        if device_raw not in set(Device):
            raise ConfigError(f"clip.device must be one of {[d.value for d in Device]}, got {device_raw!r}")
        clip = ClipConfig(
            model_name=require_field(clip_raw, "model_name", str),
            device=Device(device_raw),
            image_batch_size=require_field(clip_raw, "image_batch_size", int),
            text_batch_size=require_field(clip_raw, "text_batch_size", int),
        )
        if not clip.model_name:
            raise ConfigError("clip.model_name must not be empty")
        if clip.image_batch_size < 1 or clip.text_batch_size < 1:
            raise ConfigError("clip.image_batch_size and clip.text_batch_size must be >= 1")
        return cls(
            data_config=Path(require_field(raw, "data_config", str)),
            clip=clip,
            embeddings_dir=Path(require_field(raw, "embeddings_dir", str)),
            benchmark=_benchmark_config(raw),
            runs_dir=Path(require_field(raw, "runs_dir", str)),
            graphsage=_graphsage_config(raw),
        )

    def model_embeddings_dir(self) -> Path:
        """Directory holding the embeddings of the configured CLIP model."""
        return self.embeddings_dir / self.clip.model_name.replace("/", "--")


def _benchmark_config(raw: dict[str, Any]) -> BenchmarkConfig:
    benchmark_raw = require_section(raw, "benchmark")
    recall_ks = _number_list(benchmark_raw, "recall_ks", int)
    alpha_step = float(require_field(benchmark_raw, "alpha_step", (int, float)))
    steps = round(1.0 / alpha_step) if alpha_step > 0 else 0
    if not 0.0 < alpha_step <= 1.0 or abs(steps * alpha_step - 1.0) > _ALPHA_STEP_TOLERANCE:
        raise ConfigError(f"benchmark.alpha_step must divide 1 evenly and lie in (0, 1], got {alpha_step}")
    top_k_grid = _number_list(benchmark_raw, "top_k_grid", int)
    if any(k < 1 for k in recall_ks) or any(k < 1 for k in top_k_grid):
        raise ConfigError("benchmark.recall_ks and benchmark.top_k_grid must be >= 1")
    return BenchmarkConfig(
        recall_ks=tuple(recall_ks),
        alpha_grid=tuple(round(i / steps, _ALPHA_DECIMALS) for i in range(steps + 1)),
        top_k_grid=tuple(top_k_grid),
    )


def _number_list(raw: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> list[Any]:
    """A non-empty list of distinct numbers of the given type."""
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise ConfigError(f"benchmark.{key} must be a non-empty list")
    if not all(isinstance(v, kind) and not isinstance(v, bool) for v in values):
        raise ConfigError(f"benchmark.{key} must contain only {kind}")
    if len(set(values)) != len(values):
        raise ConfigError(f"benchmark.{key} has duplicates")
    return values


def _graphsage_config(raw: dict[str, Any]) -> GraphSageConfig:
    graphsage_raw = require_section(raw, "graphsage")
    image_node = require_field(graphsage_raw, "image_node", str)
    if image_node not in set(ImageNodeFeature):
        raise ConfigError(
            f"graphsage.image_node must be one of {[v.value for v in ImageNodeFeature]}, got {image_node!r}"
        )
    directed = graphsage_raw.get("directed")
    if not isinstance(directed, bool):
        raise ConfigError(f"graphsage.directed must be true or false, got {directed!r}")
    return GraphSageConfig(
        hidden_dim=require_field(graphsage_raw, "hidden_dim", int),
        num_layers=require_field(graphsage_raw, "num_layers", int),
        directed=directed,
        image_node=ImageNodeFeature(image_node),
    )
