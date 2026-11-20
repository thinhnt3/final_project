"""Rebuild a trained graph encoder from its run directory, for analyses run after training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.core.config_parsing import read_mapping
from src.core.exceptions import DataError
from src.data.config import DataConfig
from src.data.json_io import read_json
from src.embeddings.store import EmbeddingTable, load_table, verify_inputs_unchanged
from src.models.adamf_mat import MODALITIES, AdamfMatConfig, AdamfMatImageEncoder
from src.models.checkpoint import SETTINGS_FILE, RunDirectory
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature
from src.models.graphsage_label_text import LabelTextGraphEncoder
from src.retrieval.config import RetrievalConfig

GRAPH_FILE = "graph/image_graphs.json"


@dataclass(frozen=True)
class GraphInputs:
    """Everything a trained encoder reads, loaded once."""

    images: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    image_graphs: list[dict[str, Any]]
    image_table: EmbeddingTable
    query_table: EmbeddingTable
    concepts: torch.Tensor
    predicates: torch.Tensor

    def tables(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concept, predicate and image vectors, as the encoders take them."""
        return self.concepts, self.predicates, torch.from_numpy(self.image_table.vectors)


def _dense(table: EmbeddingTable, name: str) -> torch.Tensor:
    """Vectors of a table whose ids are 0..n-1 in order, so an id is also a row number."""
    if list(table.ids) != list(range(len(table.ids))):
        raise DataError(f"ids of the {name} embedding table are not 0..n-1 in order")
    return torch.from_numpy(table.vectors)


def load_graph_inputs(config: RetrievalConfig) -> GraphInputs:
    """Processed data and frozen CLIP tables of the configured dataset."""
    processed = DataConfig.from_yaml(config.data_config).paths.processed_dir
    embeddings_dir = config.model_embeddings_dir()
    verify_inputs_unchanged(embeddings_dir)
    return GraphInputs(
        images=read_json(processed / "images.json"),
        queries=read_json(processed / "queries.json"),
        image_graphs=read_json(processed / GRAPH_FILE),
        image_table=load_table(embeddings_dir, "images"),
        query_table=load_table(embeddings_dir, "queries"),
        concepts=_dense(load_table(embeddings_dir, "concepts"), "concepts"),
        predicates=_dense(load_table(embeddings_dir, "predicates"), "predicates"),
    )


def load_trained_encoder(run_dir: Path, method: str, inputs: GraphInputs) -> nn.Module:
    """Encoder with the architecture saved in the run's settings and its best weights, in eval mode."""
    settings = read_mapping(run_dir / SETTINGS_FILE)
    graph_raw = settings["graphsage"]
    graphsage = GraphSageConfig(
        hidden_dim=int(graph_raw["hidden_dim"]),
        num_layers=int(graph_raw["num_layers"]),
        directed=bool(graph_raw["directed"]),
        image_node=ImageNodeFeature(graph_raw["image_node"]),
    )
    model: nn.Module
    if method == "adamf_mat":
        raw = dict(settings["adamf_mat"])
        # Runs from before the modality switches used all three, unnormalized.
        raw["modalities"] = tuple(raw.get("modalities", MODALITIES))
        model = AdamfMatImageEncoder(AdamfMatConfig(**raw), graphsage, *inputs.tables())
    elif method == "graphsage_label_text":
        model = LabelTextGraphEncoder(graphsage, int(settings["label_text_dim"]), *inputs.tables())
    elif method == "graphsage":
        model = GraphImageEncoder(graphsage, *inputs.tables())
    else:
        raise DataError(f"unknown trained method {method!r}")
    model.load_state_dict(RunDirectory(run_dir).load_best()["model"])
    return model.eval()
