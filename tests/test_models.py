import json

import numpy as np
import pandas as pd

from deepvoice.models import RESULT_COLUMNS, run


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({"a": y + rng.normal(0, 0.7, n), "b": rng.normal(0, 1, n), "label": y,
                         "speaker": rng.integers(0, 10, n)})


def test_run_writes_config_and_results(tmp_path):
    train, dev = _frame(400, 0), _frame(300, 1)
    for _ in range(2):  # rerunning the same config must replace its rows, not duplicate them
        run(0, "logreg", ["a", "b"], train, [("asvspoof", "dev", dev)],
                      experiment="t", results_dir=tmp_path, n_boot=20)
    csv = pd.read_csv(tmp_path / "t.csv")
    assert list(csv.columns) == RESULT_COLUMNS and len(csv) == 1
    assert 0 <= csv.loc[0, "ci_low"] <= csv.loc[0, "eer"] <= csv.loc[0, "ci_high"] < 0.5
    cfg = json.loads((tmp_path / "configs" / "t_logreg_s0.json").read_text())
    assert cfg["features"] == ["a", "b"] and cfg["seed"] == 0


def test_grouped_cv_and_random_search():
    from deepvoice.models import grouped_cv_eer, random_search

    train, dev = _frame(500, 2), _frame(300, 3)
    assert 0 <= grouped_cv_eer("logreg", {"C": 1.0}, train, ["a", "b"]) < 0.5
    res = random_search("logreg", {"C": [0.01, 0.1, 1.0, 10.0]}, train, dev, ["a", "b"],
                        n_iter=3, fixed={"max_iter": 500})
    assert len(res) == 3 and res["dev_eer"].is_monotonic_increasing
    assert {"cv_eer", "params", "fit_time"} <= set(res.columns)
    assert '"max_iter": 500' in res.loc[0, "params"]
