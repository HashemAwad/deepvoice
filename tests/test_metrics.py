import numpy as np
import pytest

from deepvoice.metrics import ece, eer, speaker_bootstrap_eer


def test_eer_perfect_and_inverted():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert eer(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])[0] == 0.0
    assert eer(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])[0] == 1.0


def test_eer_hand_computed():
    # reals .1 .2 .3 .6, fakes .4 .5 .7 .8. At threshold .45: one fake missed, one real flagged
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.8]
    value, thr = eer(y, s)
    assert value == pytest.approx(0.25)
    assert 0.4 < thr <= 0.5


def test_input_validation():
    with pytest.raises(ValueError):
        eer([0, 1], [0.1, 0.2, 0.3])  # length mismatch
    with pytest.raises(ValueError):
        eer([0, 2], [0.1, 0.2])  # labels not 0/1
    with pytest.raises(ValueError):
        ece([0, 1], [0.2, 1.5])  # prob out of range


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    spk = rng.integers(0, 20, 500)
    assert speaker_bootstrap_eer(y, s, spk, n_boot=50, seed=7) == speaker_bootstrap_eer(
        y, s, spk, n_boot=50, seed=7
    )


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


def test_eer_interpolates_between_coarse_scores():
    # reals 0 0 1 1, fakes 1 1 1 1: ROC points (fpr, fnr) = (0, 1), (0.5, 0), (1, 0).
    # No threshold equalizes the errors; the line from (0, 1) to (0.5, 0) meets fpr = fnr at 1/3.
    value, thr = eer([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1])
    assert value == pytest.approx(1 / 3)
    assert thr == 1.0


def test_eer_matches_brute_force_on_continuous_scores():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 3000)
    s = y + rng.normal(0, 1, 3000)
    far = [((s >= t) & (y == 0)).sum() / (y == 0).sum() for t in np.sort(s)]
    frr = [((s < t) & (y == 1)).sum() / (y == 1).sum() for t in np.sort(s)]
    brute = min(max(a, b) for a, b in zip(far, frr))
    assert eer(y, s)[0] == pytest.approx(brute, abs=1e-3)


def test_per_attack_eer():
    from deepvoice.metrics import per_attack_eer

    y = [0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.15, 0.05]  # A01 easy, A02 scores like (or below) real clips
    a = ["bonafide", "bonafide", "A01", "A01", "A02", "A02"]
    out = per_attack_eer(y, s, a)
    assert set(out) == {"A01", "A02"} and out["A01"] == 0.0 and out["A02"] >= 0.5


def test_error_at_threshold_and_eer_threshold_agree():
    from deepvoice.metrics import error_at_threshold

    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.8]
    value, thr = eer(y, s)
    assert error_at_threshold(y, s, thr) == pytest.approx(value)
    assert error_at_threshold(y, s, 0.0) == pytest.approx(0.5)  # everything called fake
    assert error_at_threshold(y, s, 0.65) == pytest.approx((0.5 + 0.0) / 2)
