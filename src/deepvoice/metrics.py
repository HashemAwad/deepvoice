"""Evaluation metrics. Convention everywhere: label 1 = fake, 0 = real; higher score = more fake."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def _check(labels, other, name):
    labels = np.asarray(labels)
    other = np.asarray(other)
    if labels.shape != other.shape or labels.ndim != 1:
        raise ValueError(f"labels and {name} must be 1-D and the same length")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must be 0 (real) or 1 (fake)")
    return labels, other


def eer(labels, scores):
    """Equal error rate and the threshold where it occurs. Returns (eer, threshold).

    A clip is called fake when score >= threshold. The EER is read where the straight line
    between the two ROC points around the crossing meets fpr = fnr, so models with few distinct
    scores (e.g. a shallow tree) get a value for an operating point that exists; for continuous
    scores this equals the usual nearest-point estimate. The threshold is the nearer ROC point's.
    """
    labels, scores = _check(labels, scores, "scores")
    scores = scores.astype(float)
    if labels.min() == labels.max():
        raise ValueError("EER needs both classes present")
    # drop_intermediate=False: keep every threshold so the EER point is exact
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmax(fnr <= fpr))  # first point past the crossing; point 0 has fpr=0, fnr=1
    d0, d1 = fpr[i - 1] - fnr[i - 1], fpr[i] - fnr[i]
    a = d0 / (d0 - d1)  # where the line between points i-1 and i crosses fpr = fnr
    value = fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    j = i if abs(d1) <= abs(d0) else i - 1
    return float(value), float(thr[j])


def speaker_bootstrap_eer(labels, scores, speakers, n_boot=1000, alpha=0.05, seed=0):
    """EER with a speaker-clustered bootstrap CI: resample whole speakers, not clips.

    Returns (point_eer, ci_low, ci_high). Resamples that lose a class are skipped.
    """
    labels, scores = _check(labels, scores, "scores")
    scores = scores.astype(float)
    speakers = np.asarray(speakers)
    if speakers.shape != labels.shape:
        raise ValueError("speakers must have the same length as labels")
    groups = pd.Series(np.arange(len(labels))).groupby(speakers).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(eer(labels[idx], scores[idx])[0])
    if not vals:
        raise ValueError("every bootstrap resample lost a class; too few speakers?")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return eer(labels, scores)[0], float(lo), float(hi)


def ece(labels, probs, n_bins=15):
    """Expected calibration error of P(fake): gap between mean predicted P(fake) and the
    observed fake rate, averaged over equal-width probability bins."""
    labels, probs = _check(labels, probs, "probs")
    labels = labels.astype(float)
    probs = probs.astype(float)
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError("probs must be in [0, 1]; apply a sigmoid/softmax first")
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def per_attack_eer(labels, scores, attacks):
    """EER of each attack: that attack's fake clips against all real clips. Returns {attack: eer}."""
    labels, scores = _check(labels, scores, "scores")
    attacks = np.asarray(attacks)
    real = labels == 0
    return {a: eer(labels[real | (attacks == a)], scores[real | (attacks == a)])[0]
            for a in sorted(set(attacks[labels == 1]))}
