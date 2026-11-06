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
