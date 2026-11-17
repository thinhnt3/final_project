import numpy as np
import pytest

from src.core.exceptions import DataError
from src.embeddings.checks import caption_image_alignment


def test_alignment_is_high_for_own_image_and_low_for_others():
    images = np.eye(4, dtype=np.float32)
    captions = images[[0, 1, 2, 3, 0]]  # caption i describes image own[i]
    check = caption_image_alignment(captions, images, [0, 1, 2, 3, 0], seed=0)
    assert (check.pairs, check.own_mean, check.random_mean) == (5, 1.0, 0.0)


def test_alignment_exposes_shifted_ids():
    images = np.eye(4, dtype=np.float32)
    captions = images[[1, 2, 3, 0]]  # every caption paired with the wrong image
    check = caption_image_alignment(captions, images, [0, 1, 2, 3], seed=0)
    assert check.own_mean == 0.0


def test_alignment_rejects_bad_input():
    images = np.eye(2, dtype=np.float32)
    with pytest.raises(DataError, match="image rows"):
        caption_image_alignment(images, images, [0], seed=0)
    with pytest.raises(DataError, match="at least two"):
        caption_image_alignment(images[:1], images[:1], [0], seed=0)
