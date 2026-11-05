"""Evaluation metrics. Convention everywhere: label 1 = fake, 0 = real; higher score = more fake."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def eer(labels, scores):
    """Equal error rate and the threshold where it occurs. Returns (eer, threshold)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        raise ValueError("EER needs both classes present")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def speaker_bootstrap_eer(labels, scores, speakers, n_boot=1000, alpha=0.05, seed=0):
    """EER with a speaker-clustered bootstrap CI: resample whole speakers, not clips.

    Returns (point_eer, ci_low, ci_high). Resamples that lose a class are skipped.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    speakers = np.asarray(speakers)
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
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)
