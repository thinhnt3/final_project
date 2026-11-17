"""Embedding tables on disk: a float32 matrix plus the ids of its rows, in the same order."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.exceptions import DataError
from src.data.json_io import read_json, write_json
from src.data.sources import sha256_of

META_FILE = "meta.json"
_UNIT_NORM_TOLERANCE = 1e-4


@dataclass(frozen=True)
class EmbeddingTable:
    """Row i of `vectors` is the unit-length embedding of `ids[i]`."""

    ids: tuple[int | str, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids):
            raise DataError(f"{len(self.ids)} ids but vectors have shape {self.vectors.shape}")
        if len(set(self.ids)) != len(self.ids):
            raise DataError("embedding ids are not unique")
        if self.vectors.dtype != np.float32:
            raise DataError(f"embeddings must be float32, got {self.vectors.dtype}")
        if not np.isfinite(self.vectors).all():
            raise DataError("embeddings contain NaN or infinite values")
        norms = np.linalg.norm(self.vectors, axis=1)
        if len(norms) and np.abs(norms - 1.0).max() > _UNIT_NORM_TOLERANCE:
            raise DataError("embeddings are not unit length")

    def row_of(self) -> dict[int | str, int]:
        """Id → row index."""
        return {item_id: row for row, item_id in enumerate(self.ids)}

    def select(self, ids: Sequence[int | str]) -> np.ndarray:
        """Rows for the given ids, in the given order; raise DataError on an unknown id."""
        rows = self.row_of()
        missing = [item_id for item_id in ids if item_id not in rows]
        if missing:
            raise DataError(f"{len(missing)} ids have no embedding, e.g. {missing[:5]}")
        return self.vectors[[rows[item_id] for item_id in ids]]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length; raise DataError on a zero row."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise DataError("cannot normalize a zero vector")
    return (vectors / norms).astype(np.float32)


def encode_in_batches[Item](
    items: Sequence[Item], encode: Callable[[Sequence[Item]], np.ndarray], batch_size: int
) -> np.ndarray:
    """Encode `items` batch by batch and return unit-length float32 rows in the same order."""
    if not items:
        raise DataError("nothing to encode")
    if batch_size < 1:
        raise DataError(f"batch_size must be >= 1, got {batch_size}")
    parts = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        vectors = np.asarray(encode(batch), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise DataError(f"encoder returned shape {vectors.shape} for a batch of {len(batch)}")
        parts.append(vectors)
    return l2_normalize(np.concatenate(parts))


def table_paths(directory: Path, name: str) -> tuple[Path, Path]:
    """Paths of the matrix file and the ids file of a named table."""
    return directory / f"{name}.npy", directory / f"{name}.ids.json"


def save_table(directory: Path, name: str, table: EmbeddingTable) -> None:
    """Write `<name>.npy` and `<name>.ids.json`, creating the directory."""
    vectors_path, ids_path = table_paths(directory, name)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = vectors_path.with_name(vectors_path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, table.vectors)
    tmp.replace(vectors_path)
    write_json(ids_path, list(table.ids), indent=None)


def load_table(directory: Path, name: str) -> EmbeddingTable:
    """Read a table written by `save_table`; raise DataError if a file is missing."""
    vectors_path, ids_path = table_paths(directory, name)
    if not vectors_path.is_file() or not ids_path.is_file():
        raise DataError(f"embedding table '{name}' not found in {directory}")
    return EmbeddingTable(ids=tuple(read_json(ids_path)), vectors=np.load(vectors_path))


def verify_inputs_unchanged(directory: Path) -> None:
    """Raise DataError if an input of the embeddings, or a stored table, has changed since.

    Ids alone cannot reveal this: a rebuilt dataset can keep its ids and change the content.
    Image files are not hashed here; only the JSON inputs and the stored tables are.
    """
    meta_path = directory / META_FILE
    if not meta_path.is_file():
        raise DataError(f"{meta_path} not found; compute the embeddings first")
    meta = read_json(meta_path)
    changed = [
        path for path, digest in meta["inputs_sha256"].items()
        if not Path(path).is_file() or sha256_of(Path(path)) != digest
    ]
    if changed:
        raise DataError(f"embeddings in {directory} are stale; these inputs changed: {changed}")
    altered = [
        name
        for table in meta["tables"].values()
        for name, digest in table["sha256"].items()
        if not (directory / name).is_file() or sha256_of(directory / name) != digest
    ]
    if altered:
        raise DataError(f"embedding files in {directory} differ from meta.json: {altered}")
