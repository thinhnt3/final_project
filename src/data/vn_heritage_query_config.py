"""Typed, validated view of `configs/data/vn_heritage_queries.yaml`."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.core.config_parsing import read_mapping, require_field, require_section
from src.core.exceptions import ConfigError
from src.data.mkg_w_config import QueryRules, RelationTemplate
from src.data.vn_heritage_articles import RelationScheme


@dataclass(frozen=True)
class QueryConfig:
    """Paths, split, rules and wording of the retrieval task; see the YAML comments."""

    core_dir: Path
    out_dir: Path
    image_dir: Path
    label_languages: tuple[str, ...]
    article_relation: str
    article_prefix: str
    article_schemes: tuple[RelationScheme, ...]
    seed: int
    val_fraction: float
    test_fraction: float
    rules: QueryRules
    sparse_max: int
    medium_max: int
    templates: dict[str, RelationTemplate]

    def with_split_seed(self, seed: int) -> QueryConfig:
        """Same task under another claim split, written to its own `split-<seed>` sub-directory."""
        return replace(self, seed=seed, out_dir=self.out_dir / f"split-{seed}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> QueryConfig:
        """Load and validate the YAML file; raise ConfigError on any problem."""
        raw = read_mapping(path)
        paths = require_section(raw, "paths")
        split = require_section(raw, "split")
        rules = require_section(raw, "query_rules")
        strata = require_section(raw, "strata")
        sparse_max = require_field(strata, "sparse_max", int)
        medium_max = require_field(strata, "medium_max", int)
        if not 0 <= sparse_max < medium_max:
            raise ConfigError("strata needs 0 <= sparse_max < medium_max")
        schemes = require_field(raw, "article_schemes", list)
        known = {scheme.value for scheme in RelationScheme}
        if not schemes or not set(schemes) <= known:
            raise ConfigError(f"article_schemes must be a non-empty subset of {sorted(known)}, got {schemes}")
        templates = {
            relation: RelationTemplate(relation, relation, wording.get("tail"), wording.get("head"))
            for relation, wording in require_section(raw, "templates").items()
        }
        fractions = [float(require_field(split, key, (int, float))) for key in ("val_fraction", "test_fraction")]
        if min(fractions) <= 0 or sum(fractions) >= 1:
            raise ConfigError("split.val_fraction and split.test_fraction must be > 0 and sum to less than 1")
        return cls(
            core_dir=Path(require_field(paths, "core_dir", str)),
            out_dir=Path(require_field(paths, "out_dir", str)),
            image_dir=Path(require_field(paths, "image_dir", str)),
            label_languages=tuple(require_field(raw, "label_languages", list)),
            article_relation=require_field(raw, "article_relation", str),
            article_prefix=require_field(raw, "article_prefix", str),
            article_schemes=tuple(RelationScheme(scheme) for scheme in schemes),
            seed=require_field(split, "seed", int),
            val_fraction=fractions[0],
            test_fraction=fractions[1],
            # The class rule of MKG-W is switched off (max_prior_hits 1.0): here `max_answers`
            # alone removes class-like targets.
            rules=QueryRules(
                prior_top_k=1,
                max_prior_hits=1.0,
                min_train_triples=require_field(rules, "min_train_triples", int),
                max_answers=require_field(rules, "max_answers", int),
                max_hops=require_field(rules, "max_hops", int),
            ),
            sparse_max=sparse_max,
            medium_max=medium_max,
            templates=templates,
        )
