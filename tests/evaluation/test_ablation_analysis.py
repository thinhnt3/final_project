import numpy as np
import pytest

from src.core.exceptions import DataError
from src.evaluation.ablation_analysis import paired_analysis


def test_paired_counts_and_bootstrap_keep_captions_clustered():
    full = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    variant = np.array([[2, 2, 1, 1], [2, 2, 1, 1]])
    result = paired_analysis(
        full,
        variant,
        np.array([10, 10, 20, 20]),
        repeats=2000,
    )
    assert result["delta_r1_pp"] == 0
    assert result["image_cluster_bootstrap_95ci_pp"] == [-100, 100]
    assert result["per_seed"][0] == {
        "full_correct_variant_wrong": 2,
        "variant_correct_full_wrong": 2,
        "net_full_top1_count": 0,
    }
    identical = paired_analysis(
        full,
        full,
        np.array([10, 10, 20, 20]),
    )
    assert identical["image_cluster_bootstrap_95ci_pp"] == [0, 0]
    with pytest.raises(DataError):
        paired_analysis(
            full,
            variant[:, :2],
            np.array([10, 10, 20, 20]),
        )
