#!/usr/bin/env python3
"""Export a blinded, offline human-review package; never assign validity labels.

python -m scripts.prepare_validity_review --all
python -m scripts.prepare_validity_review --per-stratum 10 --output private/reviews/validity/pilot.json
python -m scripts.prepare_validity_review --manifest existing-manifest.json --output audit.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.download import image_path
from src.data.json_io import read_json, write_json
from src.data.sources import sha256_of
from src.evaluation.explanation_quality import relation_sparsity, select_review_queries
from src.evaluation.validity_audit import make_audit_manifest, package_audit_manifest
from src.models.run_validation import validate_artifact_inputs, validate_run
from src.retrieval.config import RetrievalConfig

log = logging.getLogger(__name__)


def fidelity_cases(
    source_path: Path,
    seed: int,
    fraction: float,
    per_stratum: int | None,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Reuse stored explanation subsets for every evaluated query or a balanced sample."""
    source = read_json(source_path)
    validate_artifact_inputs(source, str(source_path))
    config = RetrievalConfig.from_yaml(source["protocol"]["retrieval_config"])
    if str(seed) not in source["seeds"] or fraction not in source["protocol"]["fractions"]:
        raise DataError("requested checkpoint/budget is not in the fidelity trace")
    validate_run(
        config.runs_dir / "graphsage" / config.graphsage.run_name() / f"seed{seed}",
        config,
        seed,
    )
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    graphs = {g["vg_image_id"]: g for g in read_json(processed / "graph/image_graphs.json")}
    concepts = {c["concept_id"]: c["text"] for c in read_json(processed / "graph/concepts.json")}
    predicates = {p["predicate_id"]: p["text"] for p in read_json(processed / "graph/predicates.json")}
    records = source["seeds"][str(seed)]["queries"]
    selected = (
        select_review_queries(records, per_stratum, sample_seed)
        if per_stratum is not None
        else sorted((q for q in records if q["status"] == "evaluated"), key=lambda q: q["query_id"])
    )
    cases = []
    for query in selected:
        image_id = query["image_id"]
        graph = graphs[image_id]
        if len(graph["triplets"]) != query["relations"]:
            raise DataError("stored explanation no longer matches the image graph")
        budget = next(b for b in query["budgets"] if b["fraction_requested"] == fraction)
        relation_sparsity(len(graph["triplets"]), budget["selected_indices"])
        photo = image_path(processed / "images", image_id)
        relations = []
        for index in budget["selected_indices"]:
            subject, predicate, obj = graph["triplets"][index]
            relations.append(
                {
                    "index": index,
                    "subject": concepts[subject],
                    "predicate": predicates[predicate],
                    "object": concepts[obj],
                }
            )
        cases.append(
            {
                "case_id": f"{seed}:{query['query_id']}",
                "query_id": query["query_id"],
                "text": query["text"],
                "image_id": image_id,
                "image_path": str(photo),
                "image_sha256": sha256_of(photo),
                "relations": relations,
            }
        )
    log.info("selected %d/%d queries; skipped cases stay excluded", len(cases), len(records))
    return cases


def run(args: argparse.Namespace) -> None:
    """Prepare a manifest/package from a trace, exported cases, or an existing manifest."""
    if args.all and args.per_stratum is not None:
        raise DataError("choose --all or --per-stratum")
    if (args.manifest or args.cases) and (args.all or args.per_stratum is not None):
        raise DataError("sampling flags apply only to fidelity input")
    output = Path(args.output)
    if args.manifest:
        manifest_path = Path(args.manifest)
        if output.resolve() == manifest_path.resolve():
            raise DataError("output must not overwrite the input manifest")
        manifest = read_json(manifest_path)
    else:
        source_path = Path(args.cases or args.fidelity or "runs/fidelity/full-joint-test.json")
        manifest_path = output.with_suffix(".manifest.json")
        if source_path.resolve() in {output.resolve(), manifest_path.resolve()}:
            raise DataError("output must not overwrite the source artifact")
        if args.cases:
            cases = read_json(source_path)["cases"]
            sampling = "existing exported cases; preserve their diagnostic sampling limitations"
        else:
            cases = fidelity_cases(source_path, args.seed, args.fraction, args.per_stratum, args.sample_seed)
            sampling = (
                f"balanced {args.per_stratum} per prediction-relevance stratum"
                if args.per_stratum is not None
                else "all evaluated queries for the checkpoint/budget; small-graph skips retained upstream"
            )
        manifest = make_audit_manifest(cases, source_path, sampling)
        write_json(manifest_path, manifest)
    package = package_audit_manifest(manifest, manifest_path)
    write_json(output, package)
    log.info("wrote %d unlabeled audit items to %s", len(package["items"]), args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--fidelity", help="fidelity trace; default: runs/fidelity/full-joint-test.json")
    sources.add_argument("--cases", help="existing validity-cases.json")
    sources.add_argument("--manifest", help="existing validity-v2 manifest; preserves saved draft IDs/hashes")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--all", action="store_true", help="all evaluated queries (also the default)")
    parser.add_argument("--per-stratum", type=int, help="optional balanced pilot; no fixed 20-query limit")
    parser.add_argument("--sample-seed", type=int, default=20261118)
    parser.add_argument("--output", default="runs/explanation-quality/validity-audit.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args)
    except (AppError, OSError, ValueError):
        log.exception("validity package preparation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
