import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from deepvoice.features import load_audio
from deepvoice.silence import silence_features, silence_table


def _clip(lead=0.5, speech=1.0, trail=0.25, sr=16000):
    rng = np.random.default_rng(0)
    t = np.arange(int(speech * sr)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    noise = lambda n: 1e-4 * rng.standard_normal(n)  # quiet background, -80 dB
    return np.concatenate([noise(int(lead * sr)), tone, noise(int(trail * sr))]).astype(np.float32)


def test_silence_features_on_known_clip():
    f = silence_features(_clip())
    assert f["duration"] == pytest.approx(1.75)
    assert f["lead_sil"] == pytest.approx(0.5, abs=0.03)
    assert f["trail_sil"] == pytest.approx(0.25, abs=0.03)
    assert f["sil_ratio"] == pytest.approx(0.75 / 1.75, abs=0.03)


def test_no_silence_and_tiny_clip():
    f = silence_features(_clip(lead=0, trail=0))
    assert f["lead_sil"] == 0 and f["trail_sil"] == 0
    assert silence_features(np.ones(100, dtype=np.float32))["duration"] == pytest.approx(100 / 16000)


def test_loader_makes_16k_mono(tmp_path):
    stereo = np.stack([_clip(), _clip()], axis=1)
    p = tmp_path / "x.wav"
    sf.write(p, stereo[::2], 8000)  # 8 kHz stereo in
    y = load_audio(p)
    assert y.ndim == 1 and y.dtype == np.float32
    assert len(y) == pytest.approx(len(stereo[::2]) * 2, abs=2)


def test_silence_table_caches(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, _clip(), 16000)
    df = pd.DataFrame({"file": ["a"], "path": [str(p)]})
    cache = tmp_path / "s.parquet"
    first = silence_table(df, cache, n_jobs=1)
    p.unlink()  # a second call must read the cache, not the audio
    assert silence_table(df, cache, n_jobs=1).equals(first)
