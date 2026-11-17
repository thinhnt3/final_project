"""Typed, validated view of `configs/data/vg_coco.yaml`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigError
from src.data.schemas import Split
from src.data.synsets import normalize_text


@dataclass(frozen=True)
class SourceSpec:
    """One raw file: where to fetch it and where it lives under `raw_dir`."""

    name: str
    url: str
    target: str
    archive_member: str | None = None


@dataclass(frozen=True)
class PathsConfig:
    """Directories and files the pipeline reads or writes."""

    raw_dir: Path
    interim_dir: Path
    manifest_file: Path
    split_file: Path
    processed_dir: Path
    graph_card_file: Path


@dataclass(frozen=True)
class JoinConfig:
    """Rules for joining VG images with COCO captions."""

    captions_per_image: int


@dataclass(frozen=True)
class NormalizeConfig:
    """Thresholds for borrowing a synset from objects with the same raw name."""

    backfill_min_count: int
    backfill_min_share: float


@dataclass(frozen=True)
class SplitConfig:
    """Split sizes and filters; see the `split` section of the YAML."""

    seed: int
    sizes: dict[Split, int]
    min_relations: int
    n_strata: int
    exclude_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, recorded in the split file for traceability."""
        return {
            "seed": self.seed,
            "sizes": {s.value: n for s, n in self.sizes.items()},
            "min_relations": self.min_relations,
            "n_strata": self.n_strata,
            "exclude_ids": list(self.exclude_ids),
        }


@dataclass(frozen=True)
class DownloadConfig:
    """Parallelism and retry policy for image downloads."""

    workers: int
    retries: int
    timeout_s: float


@dataclass(frozen=True)
class GraphConfig:
    """Vocabulary rules for the graph build; see the `graph` section of the YAML."""

    min_train_images: int
    junk_names: frozenset[str]


@dataclass(frozen=True)
class DataConfig:
    """Root config for the VG ∩ COCO build."""

    paths: PathsConfig
    sources: tuple[SourceSpec, ...]
    join: JoinConfig
    normalize: NormalizeConfig
    split: SplitConfig
    download: DownloadConfig
    graph: GraphConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError(f"{config_path}: top level must be a mapping")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DataConfig:
        """Build from an already-parsed mapping (used by tests)."""
        paths_raw = _section(raw, "paths")
        paths = PathsConfig(
            raw_dir=Path(_field(paths_raw, "raw_dir", str)),
            interim_dir=Path(_field(paths_raw, "interim_dir", str)),
            manifest_file=Path(_field(paths_raw, "manifest_file", str)),
            split_file=Path(_field(paths_raw, "split_file", str)),
            processed_dir=Path(_field(paths_raw, "processed_dir", str)),
            graph_card_file=Path(_field(paths_raw, "graph_card_file", str)),
        )
        sources_raw = raw.get("sources")
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ConfigError("'sources' must be a non-empty list")
        sources = tuple(
            SourceSpec(
                name=_field(item, "name", str),
                url=_field(item, "url", str),
                target=_field(item, "target", str),
                archive_member=item.get("archive_member"),
            )
            for item in sources_raw
        )
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate source names: {names}")
        join_raw = _section(raw, "join")
        join = JoinConfig(captions_per_image=_field(join_raw, "captions_per_image", int))
        if join.captions_per_image < 1:
            raise ConfigError("join.captions_per_image must be >= 1")
        normalize_raw = _section(raw, "normalize")
        normalize = NormalizeConfig(
            backfill_min_count=_field(normalize_raw, "backfill_min_count", int),
            backfill_min_share=float(_field(normalize_raw, "backfill_min_share", (int, float))),
        )
        if normalize.backfill_min_count < 1:
            raise ConfigError("normalize.backfill_min_count must be >= 1")
        if not 0.5 < normalize.backfill_min_share <= 1.0:
            raise ConfigError("normalize.backfill_min_share must be in (0.5, 1.0]")
        download_raw = _section(raw, "download")
        download = DownloadConfig(
            workers=_field(download_raw, "workers", int),
            retries=_field(download_raw, "retries", int),
            timeout_s=float(_field(download_raw, "timeout_s", (int, float))),
        )
        if download.workers < 1 or download.retries < 1 or download.timeout_s <= 0:
            raise ConfigError("download.workers and download.retries must be >= 1, timeout_s > 0")
        return cls(
            paths=paths,
            sources=sources,
            join=join,
            normalize=normalize,
            split=_split_config(raw),
            download=download,
            graph=_graph_config(raw),
        )

    def source(self, name: str) -> SourceSpec:
        """Return the source with this name; raise ConfigError if absent."""
        for spec in self.sources:
            if spec.name == name:
                return spec
        raise ConfigError(f"unknown source: {name}")

    def raw_path(self, name: str) -> Path:
        """Local path of the named raw file."""
        return self.paths.raw_dir / self.source(name).target


def _split_config(raw: dict[str, Any]) -> SplitConfig:
    split_raw = _section(raw, "split")
    sizes_raw = _section(split_raw, "sizes")
    if set(sizes_raw) != {s.value for s in Split}:
        raise ConfigError(f"split.sizes must have exactly {[s.value for s in Split]}, got {list(sizes_raw)}")
    sizes = {s: _field(sizes_raw, s.value, int) for s in Split}
    if any(n < 1 for n in sizes.values()):
        raise ConfigError(f"split.sizes must all be >= 1, got {sizes_raw}")
    exclude_raw = split_raw.get("exclude_ids")
    if not isinstance(exclude_raw, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in exclude_raw):
        raise ConfigError("split.exclude_ids must be a list of integers")
    config = SplitConfig(
        seed=_field(split_raw, "seed", int),
        sizes=sizes,
        min_relations=_field(split_raw, "min_relations", int),
        n_strata=_field(split_raw, "n_strata", int),
        exclude_ids=tuple(sorted(set(exclude_raw))),
    )
    if config.min_relations < 0 or config.n_strata < 1:
        raise ConfigError("split.min_relations must be >= 0 and split.n_strata >= 1")
    return config


def _graph_config(raw: dict[str, Any]) -> GraphConfig:
    graph_raw = _section(raw, "graph")
    min_train_images = _field(graph_raw, "min_train_images", int)
    if min_train_images < 1:
        raise ConfigError("graph.min_train_images must be >= 1")
    names = graph_raw.get("junk_names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ConfigError("graph.junk_names must be a list of strings")
    # Names are compared with normalized raw names, so they must already be in that form.
    not_normalized = [n for n in names if n != normalize_text(n) or not n]
    if not_normalized:
        raise ConfigError(f"graph.junk_names must be lowercase, trimmed and non-empty: {not_normalized}")
    if len(set(names)) != len(names):
        raise ConfigError("graph.junk_names has duplicates")
    return GraphConfig(min_train_images=min_train_images, junk_names=frozenset(names))


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _field(raw: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    value = raw.get(key)
    # bool is a subclass of int; reject it so `captions_per_image: true` fails.
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ConfigError(f"field '{key}' must be {kind}, got {value!r}")
    return value
