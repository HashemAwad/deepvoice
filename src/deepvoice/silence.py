"""Silence and duration features (group S): leading/trailing silence, silence ratio, duration and
frame-energy statistics from an energy-based VAD.

VAD rule: a frame is speech if its energy is within REL_DB of the clip's loudest frame. REL_DB is
a fixed constant chosen before looking at any data, so it cannot leak dev/eval information.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from deepvoice.features import SR, load_audio

FRAME = 400  # 25 ms at 16 kHz
HOP = 160  # 10 ms
REL_DB = 35.0
COLUMNS = [
    "duration",
    "lead_sil",
    "trail_sil",
    "sil_ratio",
    "energy_mean",
    "energy_std",
    "energy_p10",
    "energy_p90",
    "energy_max",
]


def frame_energy_db(y):
    """RMS energy of each 25 ms frame (10 ms hop), in dB relative to full scale."""
    if len(y) < FRAME:
        y = np.pad(y, (0, FRAME - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, FRAME)[::HOP]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-10)


def silence_features(y, sr=SR):
    """Group S features of one clip (seconds for times, dBFS for energies)."""
    e = frame_energy_db(y)
    speech = e >= e.max() - REL_DB
    first = int(np.argmax(speech))
    last = len(speech) - 1 - int(np.argmax(speech[::-1]))
    hop_s = HOP / sr
    return {
        "duration": len(y) / sr,
        "lead_sil": first * hop_s,
        "trail_sil": (len(speech) - 1 - last) * hop_s,
        "sil_ratio": float(1 - speech.mean()),
        "energy_mean": float(e.mean()),
        "energy_std": float(e.std()),
        "energy_p10": float(np.percentile(e, 10)),
        "energy_p90": float(np.percentile(e, 90)),
        "energy_max": float(e.max()),
    }


def trim(y, sr=SR):
    """Cut leading and trailing silence (same VAD rule as the features); keeps the speech span."""
    e = frame_energy_db(y)
    speech = e >= e.max() - REL_DB
    first = int(np.argmax(speech))
    last = len(speech) - 1 - int(np.argmax(speech[::-1]))
    return y[first * HOP: min(len(y), last * HOP + FRAME)]


def _one(path):
    return silence_features(load_audio(path))


def silence_table(df, cache, n_jobs=-1):
    """Silence features for every row of a protocol table, cached at `cache` (parquet).

    If the cache exists it is loaded, never recomputed. Rows keep the order of `df["file"]`.
    """
    cache = Path(cache)
    if cache.exists():
        out = pd.read_parquet(cache)
        if list(out["file"]) != list(df["file"]):
            raise ValueError(f"{cache} does not match the requested clips; delete it on purpose")
        return out
    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_one)(p) for p in df["path"])
    out = pd.concat([df[["file"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"silence features: {len(out)} clips in {time.time() - t0:.0f}s -> {cache}")
    return out
