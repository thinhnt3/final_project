"""Weights & Biases run tracking. All calls to the W&B service go through this module."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.core.exceptions import ConfigError


class WandbMode(StrEnum):
    """`online` uploads while training, `offline` writes local files to sync later, `disabled` does nothing."""

    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


@dataclass(frozen=True)
class WandbConfig:
    """Where runs are reported; see the `wandb` section of the YAML."""

    mode: WandbMode
    project: str

    def __post_init__(self) -> None:
        if not self.project:
            raise ConfigError("wandb.project must not be empty")


class WandbTracker:
    """One W&B run per training run: config at start, one row per epoch, a summary at the end."""

    def __init__(
        self,
        config: WandbConfig,
        *,
        name: str,
        group: str,
        run_config: dict[str, Any],
        directory: Path,
        run_id: str | None = None,
    ) -> None:
        # Imported here: loading wandb takes seconds and is not needed to read the config.
        import wandb

        self._run = wandb.init(
            project=config.project,
            mode=config.mode.value,
            name=name,
            group=group,
            config=run_config,
            dir=str(directory),
            id=run_id,
            resume="allow",
            reinit="finish_previous",
            # Only the settings and metrics passed in are reported: no host name, account
            # e-mail, hardware details, git remote or console output.
            settings=wandb.Settings(
                x_disable_meta=True, x_disable_machine_info=True, disable_git=True, console="off"
            ),
        )

    @property
    def run_id(self) -> str:
        """Identifier to pass back as `run_id` so a resumed training continues the same W&B run."""
        return str(self._run.id)

    def log_epoch(self, row: dict[str, Any]) -> None:
        """Report one epoch's history row, keyed by epoch number."""
        self._run.log({key: value for key, value in row.items() if key != "epoch"}, step=int(row["epoch"]))

    def finish(self, summary: dict[str, Any], *, failed: bool = False) -> None:
        """Attach final values and close the run; a failed run is marked as crashed."""
        self._run.summary.update(summary)
        self._run.finish(exit_code=1 if failed else 0)
