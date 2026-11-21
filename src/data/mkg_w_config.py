"""Typed, validated view of `configs/data/mkg_w.yaml`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.data.config import SourceSpec
from src.data.schemas import Split

_ANCHOR = "{anchor}"


@dataclass(frozen=True)
class MkgwPaths:
    """Directories and files the MKG-W build reads or writes."""

    raw_dir: Path
    manifest_file: Path
    processed_dir: Path
    split_file: Path
    card_file: Path


@dataclass(frozen=True)
class QueryRules:
    """Thresholds deciding which triples become relation queries; see the YAML comments."""

    prior_top_k: int
    max_prior_hits: float
    min_train_triples: int
    max_answers: int
    max_hops: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, recorded in the split file for traceability."""
        return dict(vars(self))


@dataclass(frozen=True)
class DirectSplit:
    """Random entity split for queries that have no triple."""

    seed: int
    fractions: dict[Split, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"seed": self.seed, "fractions": {s.value: f for s, f in self.fractions.items()}}


@dataclass(frozen=True)
class RelationTemplate:
    """Query wording of one relation; a direction without wording is None."""

    pid: str
    label: str
    tail: str | None
    head: str | None


@dataclass(frozen=True)
class MkgwConfig:
    """Root config for the MKG-W build."""

    paths: MkgwPaths
    sources: tuple[SourceSpec, ...]
    rules: QueryRules
    direct_split: DirectSplit
    templates: dict[str, RelationTemplate]

    @classmethod
    def from_yaml(cls, path: str | Path) -> MkgwConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        return cls.from_dict(read_mapping(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MkgwConfig:
        """Build from an already-parsed mapping (used by tests)."""
        paths_raw = require_section(raw, "paths")
        paths = MkgwPaths(**{
            key: Path(require_field(paths_raw, key, str))
            for key in ("raw_dir", "manifest_file", "processed_dir", "split_file", "card_file")
        })
        sources_raw = raw.get("sources")
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ConfigError("'sources' must be a non-empty list")
        sources = tuple(
            SourceSpec(
                name=require_field(item, "name", str),
                url=require_field(item, "url", str),
                target=require_field(item, "target", str),
            )
            for item in sources_raw
        )
        return cls(
            paths=paths,
            sources=sources,
            rules=_query_rules(raw),
            direct_split=_direct_split(raw),
            templates=_templates(raw),
        )

    def raw_path(self, name: str) -> Path:
        """Local path of the named raw file; raise ConfigError if the name is unknown."""
        for spec in self.sources:
            if spec.name == name:
                return self.paths.raw_dir / spec.target
        raise ConfigError(f"unknown source: {name}")


def _query_rules(raw: dict[str, Any]) -> QueryRules:
    rules_raw = require_section(raw, "query_rules")
    rules = QueryRules(
        prior_top_k=require_field(rules_raw, "prior_top_k", int),
        max_prior_hits=float(require_field(rules_raw, "max_prior_hits", (int, float))),
        min_train_triples=require_field(rules_raw, "min_train_triples", int),
        max_answers=require_field(rules_raw, "max_answers", int),
        max_hops=require_field(rules_raw, "max_hops", int),
    )
    if min(rules.prior_top_k, rules.min_train_triples, rules.max_answers, rules.max_hops) < 1:
        raise ConfigError("query_rules.prior_top_k, min_train_triples, max_answers and max_hops must be >= 1")
    if not 0.0 < rules.max_prior_hits <= 1.0:
        raise ConfigError("query_rules.max_prior_hits must be in (0, 1]")
    return rules


def _direct_split(raw: dict[str, Any]) -> DirectSplit:
    split_raw = require_section(raw, "direct_split")
    fractions_raw = require_section(split_raw, "fractions")
    if set(fractions_raw) != {s.value for s in Split}:
        raise ConfigError(f"direct_split.fractions must have exactly {[s.value for s in Split]}")
    fractions = {s: float(require_field(fractions_raw, s.value, (int, float))) for s in Split}
    if any(f <= 0 for f in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ConfigError(f"direct_split.fractions must be positive and sum to 1, got {fractions_raw}")
    return DirectSplit(seed=require_field(split_raw, "seed", int), fractions=fractions)


def _templates(raw: dict[str, Any]) -> dict[str, RelationTemplate]:
    templates_raw = require_section(raw, "templates")
    templates: dict[str, RelationTemplate] = {}
    for pid, item in templates_raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"templates.{pid} must be a mapping")
        wording: dict[str, str | None] = {}
        for direction in ("tail", "head"):
            text = item.get(direction)
            if text is not None and (not isinstance(text, str) or text.count(_ANCHOR) != 1):
                raise ConfigError(f"templates.{pid}.{direction} must contain {_ANCHOR} exactly once")
            wording[direction] = text
        if wording["tail"] is None and wording["head"] is None:
            raise ConfigError(f"templates.{pid} needs wording for at least one direction")
        templates[str(pid)] = RelationTemplate(
            pid=str(pid), label=require_field(item, "label", str), tail=wording["tail"], head=wording["head"]
        )
    return templates
