"""Model builders and the train/evaluate loop shared by rungs 0-2 (results CSV + run config)."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deepvoice.metrics import speaker_bootstrap_eer

RESULT_COLUMNS = [
    "rung", "model", "config", "seed", "dataset", "split",
    "eer", "ci_low", "ci_high", "train_time", "latency",
]


def build(name, seed, **params):
    """A fresh sklearn model by name. Class weights are balanced (the data is ~90% fake).
    `params` override the defaults."""
    p = {"class_weight": "balanced", "random_state": seed, **params}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(**{"max_iter": 2000, **p}))
    if name == "tree":
        return DecisionTreeClassifier(**p)
    if name == "rf":
        return RandomForestClassifier(**p)
    if name == "hgb":
        return HistGradientBoostingClassifier(**p)
    raise ValueError(f"unknown model {name!r}")


def run(rung, name, features, train, evals, seed=0, params=None, experiment=None,
        results_dir="results", n_boot=1000, feature_set=None):
    """Fit `name` on `train`, score every (dataset, split, frame) in `evals`, record results.

    `train` and each eval frame need the `features` columns plus `label` and `speaker`.
    Writes results/configs/<config>.json and appends one row per eval to
    results/<experiment>.csv. `feature_set` (e.g. "mfcc20_dd") is added to the model label and
    config name so runs of one model on different features stay apart.
    Returns (fitted model, list of result rows).
    """
    params = params or {}
    experiment = experiment or f"rung{rung}"
    label = f"{name}:{feature_set}" if feature_set else name
    config = f"{experiment}_{name}_{feature_set}_s{seed}" if feature_set else f"{experiment}_{name}_s{seed}"
    results_dir = Path(results_dir)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    cfg = {"rung": rung, "model": name, "feature_set": feature_set, "seed": seed, "params": params,
           "features": list(features),
           "train": f"{len(train)} clips", "n_boot": n_boot}
    (results_dir / "configs" / f"{config}.json").write_text(json.dumps(cfg, indent=2) + "\n")

    np.random.seed(seed)
    model = build(name, seed, **params)
    t0 = time.perf_counter()
    model.fit(train[features].to_numpy(), train["label"].to_numpy())
    train_time = time.perf_counter() - t0

    rows = []
    for dataset, split, df in evals:
        x = df[features].to_numpy()
        t0 = time.perf_counter()
        scores = model.predict_proba(x)[:, 1]  # P(fake): higher = more fake
        latency_ms = (time.perf_counter() - t0) / len(df) * 1000
        point, lo, hi = speaker_bootstrap_eer(
            df["label"].to_numpy(), scores, df["speaker"].to_numpy(), n_boot=n_boot, seed=seed
        )
        rows.append(dict(zip(RESULT_COLUMNS, [rung, label, config, seed, dataset, split,
                                              point, lo, hi, train_time, latency_ms])))
    out = results_dir / f"{experiment}.csv"
    new = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if out.exists():  # replace earlier rows of this exact config, keep the rest
        old = pd.read_csv(out)
        new = pd.concat([old[old["config"] != config], new], ignore_index=True)
    new.to_csv(out, index=False, float_format="%.6g")
    return model, rows


def fit_scores(name, params, train, features, frames, seed=0):
    """Fit on `train`; return P(fake) for each frame in `frames` and the fit time in seconds."""
    model = build(name, seed, **params)
    t0 = time.perf_counter()
    model.fit(train[features].to_numpy(), train["label"].to_numpy())
    fit_time = time.perf_counter() - t0
    return [model.predict_proba(f[features].to_numpy())[:, 1] for f in frames], fit_time


def grouped_cv_eer(name, params, train, features, seed=0, n_splits=5):
    """Mean EER over speaker-grouped folds of `train` (no speaker in both fit and test part)."""
    from sklearn.model_selection import GroupKFold

    from deepvoice.metrics import eer

    vals = []
    for fit_idx, test_idx in GroupKFold(n_splits=n_splits).split(train, groups=train["speaker"]):
        fit, test = train.iloc[fit_idx], train.iloc[test_idx]
        (s,), _ = fit_scores(name, params, fit, features, [test], seed)
        vals.append(eer(test["label"].to_numpy(), s)[0])
    return float(np.mean(vals))


def random_search(name, space, train, dev, features, n_iter=30, seed=0, fixed=None, cv=True):
    """Try `n_iter` settings drawn from `space`; score each on dev and by grouped CV on train.

    `fixed` params (e.g. n_jobs) are added to every setting. Returns one row per setting with
    the params (JSON), dev EER (point estimate), CV EER and fit time, sorted by dev EER.
    """
    from sklearn.model_selection import ParameterSampler

    from deepvoice.metrics import eer

    rows = []
    for i, p in enumerate(ParameterSampler(space, n_iter=n_iter, random_state=seed)):
        p = {**p, **(fixed or {})}
        (s,), fit_time = fit_scores(name, p, train, features, [dev], seed)
        row = {"model": name, "trial": i, "params": json.dumps(p, default=str),
               "dev_eer": eer(dev["label"].to_numpy(), s)[0], "fit_time": fit_time}
        if cv:
            row["cv_eer"] = grouped_cv_eer(name, p, train, features, seed)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dev_eer").reset_index(drop=True)
