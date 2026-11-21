#!/usr/bin/env python3
"""Upload the results behind the W&B report as curated runs. Reads local files only, trains nothing.

    python -m scripts.upload_report_runs                  # entity: the logged-in W&B user's default
    python -m scripts.build_wandb_report                  # then build the report on these runs

- summary runs (config role=summary, id summary-NN): one per configuration, test metrics in %
  averaged over seeds from <runs_dir>/benchmark/*.json, plus relation-query accuracy;
- train runs (role=train, id train-NN-sSEED): one per configuration and seed, the per-epoch
  <runs_dir>/<method>/<run>/seed<n>/history.json.
Booleans s1..s5 and s2_mkg_w in the config say which report section shows a run; runs 23-28 are
the graph ablation on the MKG-W caption subset, read from runs/mkg_w_captions. Runs already uploaded
(same id, tag report-v1) are skipped, so the script can be re-run after adding entries.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import wandb

from src.connectors.wandb_tracker import resolve_entity
from src.core.exceptions import AppError, DataError
from src.data.json_io import read_json
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("upload_report_runs")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
TAG = "report-v1"
SECTIONS = ("s1", "s2", "s3", "s4", "s5", "s2_mkg_w")
# Results of the second dataset live under their own runs directory.
MKG_W_RUNS_DIR = "runs/mkg_w_captions"
ABLATION_REPORT = "plans/reports/ablation-261120-0650-relation-suite.md"
METRICS = (("recall@1", "r1"), ("recall@5", "r5"), ("recall@10", "r10"), ("mrr", "mrr"))


@dataclass(frozen=True)
class ReportRun:
    """One configuration shown in the report."""

    number: str
    label: str
    level: int
    sections: tuple[str, ...]
    benchmark: str | None = None  # file stem under <runs_dir>/benchmark
    train_dir: str | None = None  # <method>/<run> under <runs_dir>, holding seed*/history.json
    extra: dict[str, Any] = field(default_factory=dict)  # numbers whose raw outputs are gone
    runs_dir: str | None = None  # read from here instead of the config's runs_dir


# Graph-only confuser acc@1 (%) on the relation queries, 5 seeds. The raw outputs of these runs
# were lost with a deleted worktree; the numbers are those of ABLATION_REPORT.
RELATION = {
    "01": {"rel_all": 28.8, "rel_contrast": 32.6},
    "17": {"rel_all": 27.7, "rel_all_masked": 27.7, "rel_contrast": 33.6, "rel_contrast_masked": 33.7},
    "19": {"rel_all": 30.0, "rel_all_masked": 28.5, "rel_contrast": 35.0, "rel_contrast_masked": 33.2},
    "20": {"rel_all": 30.0, "rel_all_masked": 28.5, "rel_contrast": 34.9, "rel_contrast_masked": 33.1},
    "21": {"rel_all": 30.0, "rel_all_masked": 28.4, "rel_contrast": 34.9, "rel_contrast_masked": 33.0},
}

_GS = "graphsage/directed-learned-l3-h512"
RUNS = (
    ReportRun("01", "CLIP ViT-B/32 (zero-shot)", 1, ("s1", "s3", "s5"), "clip"),
    ReportRun("02", "Graph lexical match", 1, ("s1",), "graph_lexical"),
    ReportRun("03", "GraphSAGE (learned image node)", 1, ("s1", "s2", "s3"), "graphsage-directed-learned-l3-h512", _GS),
    ReportRun(
        "04",
        "GraphSAGE (CLIP image node)",
        1,
        ("s1", "s3"),
        "graphsage-directed-clip_image-l3-h512",
        "graphsage/directed-clip_image-l3-h512",
    ),
    ReportRun(
        "05",
        "GraphSAGE ablation: objects only",
        1,
        ("s2",),
        "graphsage-directed-learned-l3-h512-objects_only-a0",
        f"{_GS}-objects_only-a0",
    ),
    ReportRun(
        "06",
        "GraphSAGE ablation: drop predicates",
        1,
        ("s2",),
        "graphsage-directed-learned-l3-h512-drop_predicates-a0",
        f"{_GS}-drop_predicates-a0",
    ),
    ReportRun(
        "07",
        "GraphSAGE ablation: random edges",
        1,
        ("s2",),
        "graphsage-directed-learned-l3-h512-random_edges-a0",
        f"{_GS}-random_edges-a0",
    ),
    ReportRun(
        "08",
        "GraphSAGE ablation: undirected",
        1,
        ("s2",),
        "graphsage-directed-learned-l3-h512-undirected-a0",
        f"{_GS}-undirected-a0",
    ),
    ReportRun(
        "09",
        "Linear adapter (image modality only)",
        2,
        ("s3", "s4"),
        "adamf_mat-adamf-d256-no-structure-no-text",
        "adamf_mat/adamf-d256-no-structure-no-text",
    ),
    ReportRun(
        "10", "AdaMF-MAT (full)", 2, ("s3", "s4"), "adamf_mat-adamf-mat-d256-mu1", "adamf_mat/adamf-mat-d256-mu1"
    ),
    ReportRun("11", "AdaMF w/o MAT", 2, ("s4",), "adamf_mat-adamf-d256", "adamf_mat/adamf-d256"),
    ReportRun("12", "Mean fusion + MAT", 2, ("s4",), "adamf_mat-mean-mat-d256-mu1", "adamf_mat/mean-mat-d256-mu1"),
    ReportRun("13", "Mean fusion w/o MAT", 2, ("s4",), "adamf_mat-mean-d256", "adamf_mat/mean-d256"),
    ReportRun(
        "14",
        "AdaMF w/o structure",
        2,
        ("s4",),
        "adamf_mat-adamf-d256-no-structure",
        "adamf_mat/adamf-d256-no-structure",
    ),
    ReportRun("15", "AdaMF w/o text", 2, ("s4",), "adamf_mat-adamf-d256-no-text", "adamf_mat/adamf-d256-no-text"),
    ReportRun("16", "AdaMF w/o image", 2, ("s4",), "adamf_mat-adamf-d256-no-image", "adamf_mat/adamf-d256-no-image"),
    ReportRun(
        "17", "AdaMF + L2 (baseline)", 2, ("s3", "s4", "s5"), "adamf_mat-adamf-d256-l2", "adamf_mat/adamf-d256-l2"
    ),
    ReportRun("18", "Mean fusion + L2", 2, ("s4",), "adamf_mat-mean-d256-l2", "adamf_mat/mean-d256-l2"),
    ReportRun(
        "19",
        "L2 + structure-only dropout 0.3",
        2,
        ("s3", "s4", "s5"),
        "adamf_mat-adamf-d256-l2-so0.3",
        "adamf_mat/adamf-d256-l2-so0.3",
    ),
    ReportRun("20", "L2 + structure-only loss 0.1", 2, ("s3", "s4", "s5"), extra={"test/r1": 53.1, "seeds": 5}),
    ReportRun("21", "L2 + dropout 0.3 + loss 0.1", 2, ("s5",), extra={"test/r1": 52.9, "seeds": 5}),
    ReportRun(
        "22", "L2 + same-pair hard negatives", 2, ("s4",), "adamf_mat-adamf-d256-l2-hn4", "adamf_mat/adamf-d256-l2-hn4"
    ),
    # Graph ablation on the MKG-W caption subset: summary runs only, no learning curves in the report.
    ReportRun("23", "MKG-W subset: CLIP ViT-B/32 (zero-shot)", 1, ("s2_mkg_w",), "clip", runs_dir=MKG_W_RUNS_DIR),
    *(
        ReportRun(
            number,
            f"MKG-W subset: GraphSAGE {name}",
            1,
            ("s2_mkg_w",),
            f"graphsage-directed-learned-l3-h512{suffix}",
            runs_dir=MKG_W_RUNS_DIR,
        )
        for number, name, suffix in (
            ("24", "(learned image node)", ""),
            ("25", "ablation: objects only", "-objects_only-a0"),
            ("26", "ablation: drop predicates", "-drop_predicates-a0"),
            ("27", "ablation: random edges", "-random_edges-a0"),
            ("28", "ablation: undirected", "-undirected-a0"),
        )
    ),
)
# Train runs that also appear in the learning-curve panels of sections 1 and 3.
CURVES = {"s1": {"03", "04"}, "s3": {"04", "09", "10", "17", "19"}}


def _percent(values: list[float]) -> tuple[float, float]:
    mean = round(100 * statistics.mean(values), 2)
    return mean, round(100 * statistics.stdev(values), 2) if len(values) > 1 else 0.0


def benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Test metrics in %, mean and std over seeds; CLIP and lexical matching have no seeds."""
    out: dict[str, Any] = {}
    if "seeds" not in benchmark:
        block = benchmark["clip"] if benchmark.get("method") == "clip" else benchmark["reranked"]
        out.update({f"test/{short}": round(100 * block["test"][key], 2) for key, short in METRICS})
        if "graph_only" in benchmark:
            out["graph_only/r1"] = round(100 * benchmark["graph_only"]["test"]["recall@1"], 2)
            out["graph_only/r10"] = round(100 * benchmark["graph_only"]["test"]["recall@10"], 2)
        return out
    seeds = list(benchmark["seeds"].values())
    out["seeds"] = len(seeds)
    for mode, prefix in (("reranked", "test"), ("graph_only", "graph_only")):
        for key, short in METRICS:
            out[f"{prefix}/{short}"], out[f"{prefix}/{short}_std"] = _percent([s[mode]["test"][key] for s in seeds])
    out["alpha"] = round(statistics.mean(s["selected_on_validation"]["alpha"] for s in seeds), 3)
    return out


def history_row(row: dict[str, Any]) -> dict[str, float]:
    """Numeric fields of one epoch, retrieval metrics in % like the report."""
    numeric = {k: v for k, v in row.items() if isinstance(v, int | float) and not isinstance(v, bool)}
    return {k: 100 * v if "recall" in k or k == "val_mrr" else v for k, v in numeric.items()}


def upload(config: RetrievalConfig, entity: str, project: str) -> None:
    """Create every missing summary and train run of RUNS in entity/project."""
    api = wandb.Api(timeout=60)
    done = {r.id for r in api.runs(f"{entity}/{project}", filters={"tags": TAG})}
    for spec in RUNS:
        base = {"level": spec.level, "label": spec.label, "number": spec.number}
        runs_dir = Path(spec.runs_dir) if spec.runs_dir else config.runs_dir
        if f"summary-{spec.number}" not in done:
            summary = (
                benchmark_summary(read_json(runs_dir / "benchmark" / f"{spec.benchmark}.json"))
                if spec.benchmark
                else {}
            )
            summary.update(spec.extra)
            summary.update(RELATION.get(spec.number, {}))
            source = f"{runs_dir}/benchmark/{spec.benchmark}.json" if spec.benchmark else ABLATION_REPORT
            run = wandb.init(
                entity=entity,
                project=project,
                id=f"summary-{spec.number}",
                name=f"{spec.number} · {spec.label}",
                config={**base, "role": "summary", "source": source, **{s: s in spec.sections for s in SECTIONS}},
                tags=[TAG],
                job_type="summary",
                reinit="create_new",
            )
            run.summary.update(summary)
            run.finish()
            log.info("uploaded summary %s", spec.number)
        if spec.train_dir is None:
            continue
        seed_dirs = sorted((runs_dir / spec.train_dir).glob("seed*"))
        if not seed_dirs:
            raise DataError(f"no seed directories under {runs_dir / spec.train_dir}")
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.removeprefix("seed"))
            run_id = f"train-{spec.number}-s{seed}"
            if run_id in done:
                continue
            result = read_json(seed_dir / "result.json")
            run = wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                name=f"{spec.number} · {spec.label} · seed {seed}",
                config={
                    **base,
                    "role": "train",
                    "seed": seed,
                    "source": f"runs/{spec.train_dir}/{seed_dir.name}/history.json",
                    **{s: spec.number in CURVES.get(s, set()) for s in SECTIONS},
                },
                tags=[TAG],
                job_type="train",
                group=spec.label,
                reinit="create_new",
            )
            for row in read_json(seed_dir / "history.json"):
                run.log(history_row(row), step=int(row["epoch"]))
            run.summary.update({"best_epoch": result.get("best_epoch"), "epochs_run": result.get("epochs_run")})
            run.finish()
            log.info("uploaded %s", run_id)


def main() -> int:
    """Parse the command line and upload the missing runs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--entity", help="W&B entity; default: the logged-in user's default entity")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        upload(config, resolve_entity(args.entity), config.wandb.project)
    except AppError:
        log.exception("upload failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
