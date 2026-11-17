"""Sanity checks on stored embeddings."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.core.exceptions import DataError


@dataclass(frozen=True)
class AlignmentCheck:
    """Mean cosine of each caption with its own image versus with a random other image."""

    pairs: int
    own_mean: float
    random_mean: float


def caption_image_alignment(
    caption_vectors: np.ndarray, image_vectors: np.ndarray, image_row_of_caption: Sequence[int], seed: int
) -> AlignmentCheck:
    """Compare caption↔own-image cosine with caption↔random-other-image cosine.

    A correct image/caption pairing gives a clearly higher own mean; equal means point to
    misaligned ids. Rows must be unit length.
    """
    own_rows = np.asarray(image_row_of_caption)
    n_images = image_vectors.shape[0]
    if len(own_rows) != caption_vectors.shape[0]:
        raise DataError(f"{caption_vectors.shape[0]} captions but {len(own_rows)} image rows")
    if n_images < 2:
        raise DataError("alignment check needs at least two images")
    rng = np.random.default_rng(seed)
    # A non-zero shift modulo the image count never lands on the caption's own image.
    other_rows = (own_rows + rng.integers(1, n_images, size=len(own_rows))) % n_images
    own = np.einsum("ij,ij->i", caption_vectors, image_vectors[own_rows])
    other = np.einsum("ij,ij->i", caption_vectors, image_vectors[other_rows])
    return AlignmentCheck(
        pairs=len(own_rows), own_mean=round(float(own.mean()), 4), random_mean=round(float(other.mean()), 4)
    )
