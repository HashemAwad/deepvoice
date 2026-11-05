import numpy as np
import pytest

from deepvoice.metrics import ece, eer, speaker_bootstrap_eer


def test_eer_perfect_and_inverted():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert eer(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])[0] == 0.0
    assert eer(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])[0] == 1.0


def test_eer_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    assert abs(eer(y, rng.random(20000))[0] - 0.5) < 0.02


def test_eer_needs_both_classes():
    with pytest.raises(ValueError):
        eer([1, 1, 1], [0.1, 0.2, 0.3])


def test_bootstrap_ci_brackets_point_and_is_speaker_clustered():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1, n)  # imperfect detector
    spk = rng.integers(0, 40, n)
    point, lo, hi = speaker_bootstrap_eer(y, s, spk, n_boot=200)
    assert lo <= point <= hi and 0 < lo < hi < 1


def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(2)
    p = rng.random(50000)
    calibrated = (rng.random(50000) < p).astype(int)
    assert ece(calibrated, p) < 0.02
    always_fake = np.ones(50000)
    assert ece(np.zeros(50000), always_fake) == 1.0
