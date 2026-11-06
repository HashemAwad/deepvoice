import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from deepvoice.features import (
    GROUPS,
    N_CEP,
    STATS,
    column_names,
    feature_tables,
    frame_features,
    select,
)


def _tone(freq, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shapes_and_column_names():
    f = frame_features(_tone(440))
    assert f["mfcc"].shape[0] == f["lfcc"].shape[0] == 3 * N_CEP
    assert len(column_names("mfcc")) == 3 * N_CEP * len(STATS)
    assert len(column_names("spec")) == 6 * len(STATS)


def test_high_frequency_ratio_follows_pitch():
    hf = lambda y: frame_features(y)["spec"][5].mean()
    assert hf(_tone(6000)) > 0.9 and hf(_tone(300)) < 0.1


def test_short_clip_does_not_crash():
    f = frame_features(_tone(440, seconds=0.02))
    assert np.isfinite(f["mfcc"]).all()


def test_cache_and_ablation_subset(tmp_path):
    for i, freq in enumerate([300, 3000]):
        sf.write(tmp_path / f"{i}.wav", _tone(freq), 16000)
    df = pd.DataFrame({"file": ["0", "1"], "path": [str(tmp_path / f"{i}.wav") for i in range(2)]})
    out = feature_tables(df, "t", root=tmp_path / "f", n_jobs=1)
    assert set(out) == set(GROUPS) and len(out["lfcc"]) == 2
    (tmp_path / "0.wav").unlink()  # second call must load the cache
    again = feature_tables(df, "t", root=tmp_path / "f", n_jobs=1)
    assert again["mfcc"].equals(out["mfcc"])
    sub = select(out["mfcc"], "mfcc", n_coef=20, deltas=0)
    assert sub.shape == (2, 40)
    with pytest.raises(KeyError):
        select(out["mfcc"], "mfcc", n_coef=41)
