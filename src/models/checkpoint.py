"""Files of one training run: checkpoints, history and settings, written atomically."""
from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.core.config_parsing import read_mapping
from src.core.exceptions import TrainingError
from src.data.json_io import read_json, write_json

LAST_FILE = "last.pt"
BEST_FILE = "best.pt"
HISTORY_FILE = "history.json"
RESULT_FILE = "result.json"
SETTINGS_FILE = "config.yaml"
LOG_FILE = "train.log"
GRAPH_VECTORS = "graph_vectors"
TRACKER_ID_FILE = "tracker_run_id.txt"
TRACKER_DIR = "wandb"
# Everything a run may leave behind; removed when a run directory is started again.
_RUN_ENTRIES = (
    LAST_FILE, BEST_FILE, HISTORY_FILE, RESULT_FILE, SETTINGS_FILE, LOG_FILE,
    f"{GRAPH_VECTORS}.npy", f"{GRAPH_VECTORS}.ids.json", TRACKER_ID_FILE, TRACKER_DIR,
)
# Files whose presence means the directory already holds training progress.
_PROGRESS_FILES = (LAST_FILE, BEST_FILE, HISTORY_FILE, RESULT_FILE)
# Settings that may change between a run and its continuation without changing its results so far.
RESUMABLE_TRAINING_KEYS = frozenset({"max_epochs", "patience", "seeds"})
# Settings added after some runs were trained, with the value those older runs implicitly used.
LEGACY_ADAMF_MAT = {
    "modalities": ["structure", "image", "text"],
    "normalize_modalities": False,
    "structure_only_rate": 0.0,
    "structure_loss_weight": 0.0,
}
LEGACY_TRAINING = {
    "hard_negative_group": 1,
    "predicate_swap_weight": 0.0,
    "predicate_swap_pool": 50,
    "train_fraction": 1.0,
}


class RunDirectory:
    """Where a run keeps `last.pt` (to resume), `best.pt` (best on validation) and its history."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def has_progress(self) -> bool:
        """True if a run was started here and wrote at least one epoch or a result."""
        return any((self.path / name).is_file() for name in _PROGRESS_FILES)

    def is_finished(self) -> bool:
        """True if the run wrote its final result."""
        return (self.path / RESULT_FILE).is_file()

    def validate_finished(
        self,
        settings: Mapping[str, Any],
    ) -> None:
        """A result marker alone does not make a finished run reusable."""
        if not self.is_finished():
            raise TrainingError(f"{self.path}: no completed result")

        # Validate the snapshot before trusting the completed-run marker.
        changed = changed_settings(read_mapping(self.path / SETTINGS_FILE), settings)
        if changed:
            raise TrainingError(f"cannot skip finished run {self.path}: settings changed: {changed}")

        # Check that local checkpoints and exported vectors still exist.
        required = (BEST_FILE, LAST_FILE, f"{GRAPH_VECTORS}.npy", f"{GRAPH_VECTORS}.ids.json")
        missing = [entry for entry in required if not (self.path / entry).is_file()]
        if missing:
            raise TrainingError(f"finished run {self.path} is missing {missing}; restore artifacts or use --overwrite")

    def prepare(self, *, resume: bool, overwrite: bool) -> None:
        """Create the directory for a new run, clear it for a restart, or check it can be resumed."""
        if resume and overwrite:
            raise TrainingError("a run is either resumed or started again, not both")
        if resume and not (self.path / LAST_FILE).is_file():
            raise TrainingError(f"cannot resume: {self.path / LAST_FILE} not found")
        if self.has_progress() and not resume and not overwrite:
            raise TrainingError(f"{self.path} already holds a run; pass --resume or --overwrite")
        if not resume:
            # Leftovers of an earlier run must not be mistaken for results of the new one.
            for name in _RUN_ENTRIES:
                entry = self.path / name
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink(missing_ok=True)
        self.path.mkdir(parents=True, exist_ok=True)

    def save_last(self, state: dict[str, Any]) -> None:
        """Checkpoint of the last finished epoch."""
        self._save(LAST_FILE, state)

    def save_best(self, state: dict[str, Any]) -> None:
        """Checkpoint of the best epoch so far."""
        self._save(BEST_FILE, state)

    def load_last(self) -> dict[str, Any]:
        """State saved by `save_last`."""
        return self._load(LAST_FILE)

    def load_best(self) -> dict[str, Any]:
        """State saved by `save_best`."""
        return self._load(BEST_FILE)

    def save_history(self, history: list[dict[str, Any]]) -> None:
        """All epochs so far, one row each."""
        write_json(self.path / HISTORY_FILE, history)

    def load_history(self) -> list[dict[str, Any]]:
        """Rows written by `save_history`."""
        rows: list[dict[str, Any]] = read_json(self.path / HISTORY_FILE)
        return rows

    def _save(self, name: str, state: dict[str, Any]) -> None:
        # Write next to the target, then rename: an interrupted save never leaves a broken file.
        tmp = self.path / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(self.path / name)

    def _load(self, name: str) -> dict[str, Any]:
        file = self.path / name
        if not file.is_file():
            raise TrainingError(f"checkpoint not found: {file}")
        state: dict[str, Any] = torch.load(file, map_location="cpu", weights_only=True)
        return state


def changed_settings(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Names of the settings that differ between a run's saved snapshot and the current one.

    Compared: model architecture, CLIP model, input hashes and every training setting except
    those that only decide how long training goes on. A continuation with any other change
    would not reproduce an uninterrupted run.
    """
    top_level = ("graphsage", "adamf_mat", "clip_model", "inputs_sha256", "seed")

    # Legacy snapshots implicitly used the full graph and topology seed zero.
    def normalized(
        key: str,
        settings: Mapping[str, Any],
    ) -> Any:
        value = settings.get(key)
        if key == "graphsage" and isinstance(value, dict):
            return {"ablation": "full", "ablation_seed": 0, **value}
        if key == "adamf_mat" and isinstance(value, dict):
            return {**LEGACY_ADAMF_MAT, **value}
        return value

    changed = [key for key in top_level if normalized(key, saved) != normalized(key, current)]
    saved_training = {**LEGACY_TRAINING, **saved.get("training", {})}
    current_training = {**LEGACY_TRAINING, **current.get("training", {})}
    changed += [
        f"training.{key}"
        for key in sorted(set(saved_training) | set(current_training))
        if key not in RESUMABLE_TRAINING_KEYS and saved_training.get(key) != current_training.get(key)
    ]
    return changed
