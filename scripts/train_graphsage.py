#!/usr/bin/env python3
"""Train a graph encoder (GraphSAGE or AdaMF-MAT) on VG ∩ COCO, one run per seed.

Run from the repository root after `embed_vg_coco`:
    python -m scripts.train_graphsage                 # every seed in the config
    python -m scripts.train_graphsage --seed 13       # one seed
    python -m scripts.train_graphsage --resume        # finish what is unfinished
    python -m scripts.train_graphsage --image-node clip_image   # variant; overrides the config
    python -m scripts.train_graphsage --model adamf_mat   # AdaMF-MAT encoder (Zhang et al., COLING 2024)
    python -m scripts.train_graphsage --model adamf_mat --no-mat   # an ablation

Each run writes to <runs_dir>/<model>/<name>/seed<seed>/: config.yaml, train.log,
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
from enum import StrEnum
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from src.connectors.wandb_tracker import WandbMode, WandbTracker
from src.core.config_parsing import read_mapping
from src.core.exceptions import AppError, DataError, TrainingError
from src.data.config import DataConfig
from src.data.json_io import read_json, write_json
from src.data.schemas import Split
from src.data.sources import sha256_of
from src.data.yaml_io import write_yaml
from src.embeddings.store import META_FILE, EmbeddingTable, load_table, save_table, verify_inputs_unchanged
from src.models.adamf_mat import MODALITIES, AdamfMatImageEncoder
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
from src.models.graphsage_label_text import LabelTextGraphEncoder
from src.models.train import RetrievalSet, encode_graphs, fit, info_nce, seed_everything
from src.models.train_data import build_retrieval_set, hard_negative_partners
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("train_graphsage")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
GRAPH_FILE = "graph/image_graphs.json"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SETTINGS_HEADER = "Settings of this training run. Written by train_graphsage."
UNKNOWN_COMMIT = "unknown"
FAILURE_EXIT_CODE = 1
INTERRUPTED_EXIT_CODE = 130
_METRIC_DECIMALS = 4
# Hard-negative candidates kept per train image; a group draws from them.
HARD_NEGATIVE_PARTNERS = 32


class Encoder(StrEnum):
    """Graph encoder to train; also the sub-directory of its runs."""

    GRAPHSAGE = "graphsage"
    ADAMF_MAT = "adamf_mat"
    # GraphSAGE plus AdaMF-MAT's label-text modality, concatenated: a control for AdaMF-MAT.
    GRAPHSAGE_LABEL_TEXT = "graphsage_label_text"

    def default_run_name(self, config: RetrievalConfig) -> str:
        """Run name derived from the configured architecture of this encoder and its batching."""
        if self is Encoder.ADAMF_MAT:
            name = config.adamf_mat.run_name()
        elif self is Encoder.GRAPHSAGE_LABEL_TEXT:
            name = f"{config.graphsage.run_name()}-labeltext-d{config.adamf_mat.dim}"
        else:
            name = config.graphsage.run_name()
        group = config.training.hard_negative_group
        return f"{name}-hn{group}" if group > 1 else name


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


def _smoke_step(model: nn.Module, train: RetrievalSet, config: RetrievalConfig) -> None:
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


def build_model(config: RetrievalConfig, encoder: Encoder, data: dict[str, Any]) -> nn.Module:
    """A freshly initialized encoder of the requested kind."""
    tables = (data["concepts"], data["predicates"], data["images"])
    if encoder is Encoder.ADAMF_MAT:
        return AdamfMatImageEncoder(config.adamf_mat, config.graphsage, *tables)
    if encoder is Encoder.GRAPHSAGE_LABEL_TEXT:
        # Text branch of the same size as AdaMF-MAT's modalities, so the control matches it.
        return LabelTextGraphEncoder(config.graphsage, config.adamf_mat.dim, *tables)
    return GraphImageEncoder(config.graphsage, *tables)


def _settings(
    config: RetrievalConfig, config_path: str, name: str, seed: int, data: dict[str, Any], encoder: Encoder
) -> dict[str, Any]:
    """Everything that defines this run, for the snapshot file and the tracker."""
    return {
        **(
            {"adamf_mat": {**asdict(config.adamf_mat), "modalities": list(config.adamf_mat.modalities)}}
            if encoder is Encoder.ADAMF_MAT
            else {}
        ),
        **({"label_text_dim": config.adamf_mat.dim} if encoder is Encoder.GRAPHSAGE_LABEL_TEXT else {}),
        "run_name": name,
        "seed": seed,
        "config_file": config_path,
        "git": _git_state(),
        "clip_model": config.clip.model_name,
        "graphsage": {**asdict(config.graphsage), "image_node": config.graphsage.image_node.value},
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
    encoder: Encoder,
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
    model = build_model(config, encoder, data)
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
    encoder: Encoder,
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
    image_graphs = read_json(processed / GRAPH_FILE)
    image_table, query_table = load_table(embeddings_dir, "images"), load_table(embeddings_dir, "queries")

    train, train_ids = build_retrieval_set(Split.TRAIN, images, queries, image_graphs, image_table, query_table)
    if config.training.hard_negative_group > 1:
        partners = hard_negative_partners(image_graphs, train_ids, HARD_NEGATIVE_PARTNERS)
        train = replace(train, partners=partners)
        log.info("hard negatives: %d of %d train images have partners",
                 sum(1 for p in partners if p), len(partners))
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
        run_dir = RunDirectory(config.runs_dir / encoder.value / name / f"seed{seed}")
        if resume and run_dir.is_finished():
            log.info("seed %d already finished in %s; skipped", seed, run_dir.path)
            continue
        resume_seed = resume and run_dir.has_progress()
        run_dir.prepare(resume=resume_seed, overwrite=overwrite)
        handler = logging.FileHandler(run_dir.path / LOG_FILE)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        handlers.append(handler)
        settings = _settings(config, config_path, name, seed, data, encoder)
        _train_seed(config, name, seed, data, settings, run_dir, encoder, resume=resume_seed)
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
    parser.add_argument("--model", choices=[e.value for e in Encoder], default=Encoder.GRAPHSAGE.value)
    parser.add_argument("--no-adaptive-fusion", action="store_true", help="adamf_mat: fuse the modalities by a mean")
    parser.add_argument("--no-mat", action="store_true", help="adamf_mat: train without the generator")
    parser.add_argument("--mu", type=float, help="overrides adamf_mat.mu")
    parser.add_argument(
        "--drop-modality",
        action="append",
        choices=list(MODALITIES),
        default=[],
        help="adamf_mat: leave this modality out of the fusion; repeatable; needs --no-mat",
    )
    parser.add_argument(
        "--normalize-modalities", action="store_true", help="adamf_mat: L2-normalize each modality before fusion"
    )
    parser.add_argument(
        "--structure-only-rate", type=float, help="adamf_mat: share of train images fused from structure alone"
    )
    parser.add_argument(
        "--hard-negative-group", type=int, help="images per hard-negative group in a batch; overrides the config"
    )
    parser.add_argument("--wandb-mode", choices=[m.value for m in WandbMode], help="overrides wandb.mode")
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
        adamf_mat = replace(
            config.adamf_mat,
            adaptive_fusion=config.adamf_mat.adaptive_fusion and not args.no_adaptive_fusion,
            mat=config.adamf_mat.mat and not args.no_mat,
            mu=config.adamf_mat.mu if args.mu is None else args.mu,
            modalities=tuple(m for m in config.adamf_mat.modalities if m not in args.drop_modality),
            normalize_modalities=config.adamf_mat.normalize_modalities or args.normalize_modalities,
            structure_only_rate=(
                config.adamf_mat.structure_only_rate if args.structure_only_rate is None else args.structure_only_rate
            ),
        )
        config = replace(config, adamf_mat=adamf_mat)
        if args.hard_negative_group is not None:
            config = replace(config, training=replace(config.training, hard_negative_group=args.hard_negative_group))
        if args.wandb_mode:
            config = replace(config, wandb=replace(config.wandb, mode=WandbMode(args.wandb_mode)))
        encoder = Encoder(args.model)
        seeds = args.seed or list(config.training.seeds)
        run(
            config, args.config, args.name or encoder.default_run_name(config), seeds, handlers, encoder,
            resume=args.resume, overwrite=args.overwrite,
        )
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
