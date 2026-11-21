"""Typed view of the `adamf_mat` section of a retrieval YAML."""
from __future__ import annotations

from typing import Any

from src.core.config_parsing import require_field, require_section
from src.core.exceptions import ConfigError
from src.embeddings.device import Device
from src.models.kg_adamf_mat import KgAdamfMatConfig
from src.models.kg_adamf_train import KgAdamfTrainingConfig

_NUMBER = (int, float)


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"adamf_mat.{key} must be true or false, got {value!r}")
    return value


def load_adamf_configs(raw: dict[str, Any]) -> tuple[KgAdamfMatConfig, KgAdamfTrainingConfig]:
    """Model and optimization settings of AdaMF-MAT from a parsed retrieval config."""
    section = require_section(raw, "adamf_mat")
    device = require_field(section, "device", str)
    if device not in {d.value for d in Device}:
        raise ConfigError(f"adamf_mat.device must be one of {[d.value for d in Device]}, got {device!r}")
    model = KgAdamfMatConfig(
        dim=require_field(section, "dim", int),
        margin=float(require_field(section, "margin", _NUMBER)),
        epsilon=float(require_field(section, "epsilon", _NUMBER)),
        negatives=require_field(section, "negatives", int),
        adversarial_temperature=float(require_field(section, "adversarial_temperature", _NUMBER)),
        adaptive_fusion=_require_bool(section, "adaptive_fusion"),
        mat=_require_bool(section, "mat"),
        mu=float(require_field(section, "mu", _NUMBER)),
        noise_dim=require_field(section, "noise_dim", int),
        generator_hidden_dim=require_field(section, "generator_hidden_dim", int),
        generator_learning_rate=float(require_field(section, "generator_learning_rate", _NUMBER)),
    )
    training = KgAdamfTrainingConfig(
        device=Device(device),
        batch_size=require_field(section, "batch_size", int),
        learning_rate=float(require_field(section, "learning_rate", _NUMBER)),
        max_epochs=require_field(section, "max_epochs", int),
        eval_every=require_field(section, "eval_every", int),
        patience=require_field(section, "patience", int),
        eval_batch_size=require_field(section, "eval_batch_size", int),
    )
    return model, training
