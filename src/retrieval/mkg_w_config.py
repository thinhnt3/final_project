"""Typed, validated view of `configs/retrieval/mkg_w.yaml`."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.kg_adamf_mat import KgAdamfMatConfig
from src.models.kg_adamf_train import KgAdamfTrainingConfig
from src.models.kg_graphsage import KgSageConfig
from src.models.relation_paths import RelationPathConfig
from src.models.train import TrainingConfig
from src.retrieval.config import ClipConfig, _training_config


class SelectionQueries(StrEnum):
    """Which validation queries pick the best epoch of a graph model."""

    ALL = "all"
    INDIRECT = "indirect"


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
    anchor_self_weight_grid: tuple[float, ...]
    graphsage: KgSageConfig
    selection_queries: SelectionQueries
    relation_paths: RelationPathConfig
    training: TrainingConfig
    path_training: TrainingConfig
    adamf_mat: KgAdamfMatConfig
    adamf_training: KgAdamfTrainingConfig
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
        self_weights = require_section(raw, "anchor_linking").get("self_weight_grid")
        if (not isinstance(self_weights, list) or not self_weights
                or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and w >= 0 for w in self_weights)):
            raise ConfigError("anchor_linking.self_weight_grid must be a non-empty list of numbers >= 0")
        sage_raw = require_section(raw, "kg_graphsage")
        selection = require_field(sage_raw, "selection_queries", str)
        if selection not in {s.value for s in SelectionQueries}:
            raise ConfigError(f"kg_graphsage.selection_queries must be one of {[s.value for s in SelectionQueries]}")
        paths_raw = require_section(raw, "relation_paths")
        training = _training_config(raw)
        adamf_raw = require_section(raw, "adamf_mat")
        adamf_device = require_field(adamf_raw, "device", str)
        if adamf_device not in {d.value for d in Device}:
            raise ConfigError(f"adamf_mat.device must be one of {[d.value for d in Device]}, got {adamf_device!r}")
        number = (int, float)
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
            anchor_self_weight_grid=tuple(float(w) for w in self_weights),
            graphsage=KgSageConfig(
                hidden_dim=require_field(sage_raw, "hidden_dim", int),
                num_layers=require_field(sage_raw, "num_layers", int),
                structure_only_rate=float(require_field(sage_raw, "structure_only_rate", (int, float))),
            ),
            selection_queries=SelectionQueries(selection),
            relation_paths=RelationPathConfig(max_hops=require_field(paths_raw, "max_hops", int)),
            training=training,
            # Same seeds, patience and stopping rule; only the optimizer step differs.
            path_training=replace(
                training,
                batch_size=require_field(paths_raw, "batch_size", int),
                learning_rate=float(require_field(paths_raw, "learning_rate", (int, float))),
            ),
            adamf_mat=KgAdamfMatConfig(
                dim=require_field(adamf_raw, "dim", int),
                margin=float(require_field(adamf_raw, "margin", number)),
                epsilon=float(require_field(adamf_raw, "epsilon", number)),
                negatives=require_field(adamf_raw, "negatives", int),
                adversarial_temperature=float(require_field(adamf_raw, "adversarial_temperature", number)),
                adaptive_fusion=require_field(adamf_raw, "adaptive_fusion", bool),
                mat=require_field(adamf_raw, "mat", bool),
                mu=float(require_field(adamf_raw, "mu", number)),
                noise_dim=require_field(adamf_raw, "noise_dim", int),
                generator_hidden_dim=require_field(adamf_raw, "generator_hidden_dim", int),
                generator_learning_rate=float(require_field(adamf_raw, "generator_learning_rate", number)),
            ),
            adamf_training=KgAdamfTrainingConfig(
                device=Device(adamf_device),
                batch_size=require_field(adamf_raw, "batch_size", int),
                learning_rate=float(require_field(adamf_raw, "learning_rate", number)),
                max_epochs=require_field(adamf_raw, "max_epochs", int),
                eval_every=require_field(adamf_raw, "eval_every", int),
                patience=require_field(adamf_raw, "patience", int),
                eval_batch_size=require_field(adamf_raw, "eval_batch_size", int),
            ),
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
