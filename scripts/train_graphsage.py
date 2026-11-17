#!/usr/bin/env python3
"""Train the GraphSAGE graph encoder on VG ∩ COCO, one run per seed.

Run from the repository root after `embed_vg_coco`:
    python -m scripts.train_graphsage                 # every seed in the config
    python -m scripts.train_graphsage --seed 13       # one seed
    python -m scripts.train_graphsage --resume        # finish what is unfinished
    python -m scripts.train_graphsage --image-node clip_image   # variant; overrides the config

Each run writes to <runs_dir>/graphsage/<name>/seed<seed>/: config.yaml, train.log,
history.json, last.pt, best.pt, result.json and the graph vectors of every image.
With --resume a finished seed is skipped, an interrupted one continues from last.pt and
one that never started begins from scratch. Only train and validation data are read; the
test split is scored by run_benchmark.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import asdict, replace
from typing import Any

import torch
from torch_geometric.data import Batch

from src.connectors.wandb_tracker import WandbTracker
from src.core.config_parsing import read_mapping
from src.core.exceptions import AppError, DataError, TrainingError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.data.yaml_io import write_yaml
from src.embeddings.store import META_FILE, EmbeddingTable, load_table, save_table, verify_inputs_unchanged
from src.models.ablations import Ablation, ablate_graph
from src.models.checkpoint import (
    GRAPH_VECTORS,
    LOG_FILE,
    RESULT_FILE,
    SETTINGS_FILE,
    TRACKER_ID_FILE,
    RunDirectory,
    changed_settings,
)
from src.models.graph_data import image_graph_to_data
from src.models.graphsage import GraphImageEncoder, ImageNodeFeature
from src.models.train import RetrievalSet, encode_graphs, fit, info_nce, seed_everything
from src.models.train_data import build_retrieval_set
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("train_graphsage")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
RUNS_SUBDIR = "graphsage"
GRAPH_FILE = "graph/image_graphs.json"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SETTINGS_HEADER = "Settings of this training run. Written by train_graphsage."
UNKNOWN_COMMIT = "unknown"
FAILURE_EXIT_CODE = 1
INTERRUPTED_EXIT_CODE = 130
_METRIC_DECIMALS = 4


def _git_state() -> dict[str, Any]:
    """Current commit and whether the working tree differs from it; unknown without git."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD") or UNKNOWN_COMMIT, "dirty": bool(git("status", "--porcelain"))}
    except OSError:
        return {"commit": UNKNOWN_COMMIT, "dirty": False}


def _dense_table(table: EmbeddingTable, name: str) -> torch.Tensor:
    """Vectors of a table whose ids are 0..n-1 in order, so an id is also a row number."""
    if list(table.ids) != list(range(len(table.ids))):
        raise DataError(f"ids of the {name} embedding table are not 0..n-1 in order")
    return torch.from_numpy(table.vectors)


def _smoke_step(model: GraphImageEncoder, train: RetrievalSet, config: RetrievalConfig) -> None:
    """One forward and backward pass on a single batch, without changing the weights."""
    size = min(config.training.batch_size, len(train.graphs))
    batch = Batch.from_data_list(list(train.graphs[:size])).to(config.training.device.value)
    texts = train.caption_vectors[:size, 0].to(config.training.device.value)
    loss = info_nce(model(batch), texts, config.training.temperature)
    if not torch.isfinite(loss):
        raise DataError(f"loss of the first batch is {loss.item()} before any training")
    loss.backward()
    model.zero_grad()
    log.info("smoke step ok: loss %.4f on %d images", loss.item(), size)


def _settings(config: RetrievalConfig, config_path: str, name: str, seed: int, data: dict[str, Any]) -> dict[str, Any]:
    """Everything that defines this run, for the snapshot file and the tracker."""
    return {
        "run_name": name,
        "seed": seed,
        "config_file": config_path,
        "git": _git_state(),
        "clip_model": config.clip.model_name,
        "graphsage": {
            **asdict(config.graphsage),
            "image_node": config.graphsage.image_node.value,
            "ablation": config.graphsage.ablation.value,
        },
        "training": {
            **asdict(config.training),
            "seeds": list(config.training.seeds),
            "device": config.training.device.value,
            "selection_metric": config.training.selection_metric.value,
        },
        "recall_ks": list(config.benchmark.recall_ks),
        "train_images": len(data["train"].graphs),
        "val_images": len(data["val"].graphs),
        "inputs_sha256": data["inputs_sha256"],
    }


def _train_seed(
    config: RetrievalConfig,
    name: str,
    seed: int,
    data: dict[str, Any],
    settings: dict[str, Any],
    run: RunDirectory,
    *,
    resume: bool,
) -> None:
    """Train one seed in its prepared run directory and export the graph vectors of every image."""
    log.info("run %s seed %d → %s%s", name, seed, run.path, " (resume)" if resume else "")
    if resume:
        # A continuation with other settings would not reproduce an uninterrupted run.
        changed = changed_settings(read_mapping(run.path / SETTINGS_FILE), settings)
        if changed:
            raise TrainingError(f"cannot resume {run.path}: settings changed since the run started: {changed}")
    else:
        write_yaml(run.path / SETTINGS_FILE, settings, header=SETTINGS_HEADER)

    seed_everything(seed)
    model = GraphImageEncoder(config.graphsage, data["concepts"], data["predicates"], data["images"])
    model.to(config.training.device.value)
    _smoke_step(model, data["train"], config)

    tracker_id_file = run.path / TRACKER_ID_FILE
    tracker = WandbTracker(
        config.wandb,
        name=f"{name}-seed{seed}",
        group=name,
        run_config=settings,
        directory=run.path,
        run_id=tracker_id_file.read_text().strip() if tracker_id_file.is_file() else None,
    )
    tracker_id_file.write_text(tracker.run_id)
    failed = True
    try:
        result = fit(
            model,
            data["train"],
            data["val"],
            config.training,
            seed,
            run,
            config.benchmark.recall_ks,
            resume=resume,
            show_progress=True,
            on_epoch=tracker.log_epoch,
        )
        best = run.load_best()
        model.load_state_dict(best["model"])
        vectors = encode_graphs(model, data["all_graphs"], config.training).numpy()
        save_table(run.path, GRAPH_VECTORS, EmbeddingTable(ids=data["all_image_ids"], vectors=vectors))
        summary = {
            "best_epoch": result.best_epoch,
            "epochs_run": len(result.history),
            "stop_reason": result.stop_reason.value,
            "best_validation": {key: round(value, _METRIC_DECIMALS) for key, value in best["metrics"].items()},
        }
        write_json(run.path / RESULT_FILE, summary)
        failed = False
        log.info(
            "seed %d done: best epoch %d of %d (%s), validation %s",
            seed,
            result.best_epoch,
            len(result.history),
            result.stop_reason.value,
            summary["best_validation"],
        )
    finally:
        final = (
            {}
            if failed
            else {"best_epoch": result.best_epoch, **{f"best_val_{k}": v for k, v in best["metrics"].items()}}
        )
        tracker.finish(final, failed=failed)


def run(
    config: RetrievalConfig,
    config_path: str,
    name: str,
    seeds: list[int],
    handlers: list[logging.Handler],
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    """Check the inputs, build the train and validation sets once, then train every seed.

    `handlers` receives the log-file handler of the seed in progress, so that whoever reports
    a failure can still write it to that run's log before closing the handler.
    """
    data_config = DataConfig.from_yaml(config.data_config)
    processed = data_config.paths.processed_dir
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    images = read_json(processed / "images.json")
    queries = read_json(processed / "queries.json")

    # Apply the same intervention before building train, validation and export graphs.
    image_graphs = [
        ablate_graph(
            g,
            config.graphsage.ablation,
            config.graphsage.ablation_seed,
        )
        for g in read_json(processed / GRAPH_FILE)
    ]
    image_table, query_table = load_table(embeddings_dir, "images"), load_table(embeddings_dir, "queries")

    train, _ = build_retrieval_set(Split.TRAIN, images, queries, image_graphs, image_table, query_table)
    val, _ = build_retrieval_set(Split.VAL, images, queries, image_graphs, image_table, query_table)
    graph_of = {int(g["vg_image_id"]): g for g in image_graphs}
    without_graph = [image_id for image_id in image_table.ids if int(image_id) not in graph_of]
    if without_graph:
        raise DataError(f"{len(without_graph)} embedded images have no graph, e.g. {without_graph[:5]}")
    inputs = (embeddings_dir / META_FILE, processed / GRAPH_FILE, data_config.paths.split_file)
    data = {
        "train": train,
        "val": val,
        # Sub-graphs of every image, in the order of the image table, to export graph vectors.
        "all_graphs": [image_graph_to_data(graph_of[int(i)], row) for row, i in enumerate(image_table.ids)],
        "all_image_ids": image_table.ids,
        "concepts": _dense_table(load_table(embeddings_dir, "concepts"), "concepts"),
        "predicates": _dense_table(load_table(embeddings_dir, "predicates"), "predicates"),
        "images": torch.from_numpy(image_table.vectors),
        "inputs_sha256": {str(p): sha256_of(p) for p in inputs},
    }
    log.info(
        "train %d images, validation %d images, %d captions per image",
        len(train.graphs),
        len(val.graphs),
        train.caption_vectors.shape[1],
    )
    for seed in seeds:
        run_dir = RunDirectory(config.runs_dir / RUNS_SUBDIR / name / f"seed{seed}")
        if resume and run_dir.is_finished():
            log.info("seed %d already finished in %s; skipped", seed, run_dir.path)
            continue
        resume_seed = resume and run_dir.has_progress()
        run_dir.prepare(resume=resume_seed, overwrite=overwrite)
        handler = logging.FileHandler(run_dir.path / LOG_FILE)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        handlers.append(handler)
        _train_seed(
            config, name, seed, data, _settings(config, config_path, name, seed, data), run_dir, resume=resume_seed
        )
        _close(handlers)


def _close(handlers: list[logging.Handler]) -> None:
    """Detach and close the given log handlers, emptying the list."""
    while handlers:
        handler = handlers.pop()
        logging.getLogger().removeHandler(handler)
        handler.close()


def main() -> int:
    """Parse the command line, train, and report any failure once, to the console and the run's log."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--name", help="run name; default is derived from the architecture")
    parser.add_argument("--seed", type=int, action="append", help="seed to train; repeatable; default: all in config")
    parser.add_argument(
        "--image-node",
        choices=[v.value for v in ImageNodeFeature],
        help="initial feature of the image node; overrides graphsage.image_node of the config",
    )
    parser.add_argument("--ablation", choices=[v.value for v in Ablation])
    parser.add_argument(
        "--ablation-seed",
        type=int,
        help="fixed graph randomization seed (default: config or 0)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="skip finished seeds, continue interrupted ones")
    mode.add_argument("--overwrite", action="store_true", help="delete an existing run of a seed and start again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    handlers: list[logging.Handler] = []
    try:
        config = RetrievalConfig.from_yaml(args.config)
        if args.image_node:
            model_config = replace(config.graphsage, image_node=ImageNodeFeature(args.image_node))
            config = replace(config, graphsage=model_config)
        if args.ablation:
            config = replace(
                config,
                graphsage=replace(
                    config.graphsage,
                    ablation=Ablation(args.ablation),
                ),
            )
        if args.ablation_seed is not None:
            config = replace(
                config,
                graphsage=replace(
                    config.graphsage,
                    ablation_seed=args.ablation_seed,
                ),
            )
        seeds = args.seed or list(config.training.seeds)
        name = args.name or config.graphsage.run_name()
        run(config, args.config, name, seeds, handlers, resume=args.resume, overwrite=args.overwrite)
    except AppError:
        log.exception("training failed")
        return FAILURE_EXIT_CODE
    except KeyboardInterrupt:
        log.warning("interrupted; finished epochs are saved, continue with --resume")
        return INTERRUPTED_EXIT_CODE
    except Exception:
        # Boundary of the script: errors from libraries (tracker, torch, filesystem) are logged
        # to the run's log too, then reported through the exit code.
        log.exception("training crashed")
        return FAILURE_EXIT_CODE
    finally:
        _close(handlers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
