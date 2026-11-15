"""Audio loading and features: one loader for every dataset (16 kHz mono), MFCC/LFCC/spectral
features pooled over time, cached to data/ and never recomputed.

We compute the largest setting once (40 coefficients, static + delta + delta-delta, 7 pooling
statistics). Smaller settings for the E2 ablations (13/20 coefficients, no deltas, mean+std only)
are column subsets of the same numbers: the first k DCT coefficients do not depend on how many
are kept.
"""

import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.fft import dct

SR = 16000
N_FFT, WIN, HOP = 512, 400, 160  # 32 ms FFT, 25 ms window, 10 ms hop
N_CEP = 40
N_MELS, N_LIN = 80, 70
STATS = ["mean", "std", "min", "max", "p10", "p50", "p90"]
SPECTRAL = ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "hf_ratio"]
GROUPS = ["mfcc", "lfcc", "spec"]


def load_audio(path, sr=SR):
    """The single loader for ASVspoof and In-the-Wild: float32 mono at `sr` Hz."""
    y, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y.astype(np.float32)


def _linear_filterbank(n_filters=N_LIN, n_fft=N_FFT, sr=SR):
    """Triangular filters evenly spaced on a linear Hz scale (the LFCC counterpart of mel)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))


MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
LIN_FB = _linear_filterbank()


def _cepstra(power, fb):
    """Log filterbank energies -> DCT -> first N_CEP coefficients, plus delta and delta-delta."""
    c = dct(10 * np.log10(fb @ power + 1e-10), type=2, norm="ortho", axis=0)[:N_CEP]
    d1 = librosa.feature.delta(c, order=1, mode="nearest")
    d2 = librosa.feature.delta(c, order=2, mode="nearest")
    return np.concatenate([c, d1, d2])  # (3 * N_CEP, frames)


def _pool(x):
    """(features, frames) -> (features * len(STATS),), statistic-major per feature."""
    p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=1)
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1), p10, p50, p90], axis=1).ravel()


def frame_features(y, sr=SR):
    """Frame-level MFCC, LFCC (each 40 + deltas) and spectral descriptors of one clip."""
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    power = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = power.sum(axis=0) + 1e-10
    spec = np.stack([
        librosa.feature.spectral_centroid(S=power, sr=sr, n_fft=N_FFT)[0],
        librosa.feature.spectral_bandwidth(S=power, sr=sr, n_fft=N_FFT)[0],
        librosa.feature.spectral_rolloff(S=power, sr=sr, n_fft=N_FFT, roll_percent=0.85)[0],
        librosa.feature.spectral_flatness(S=power)[0],
        librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0][: power.shape[1]],
        power[freqs >= 4000].sum(axis=0) / total,  # high-frequency energy ratio (>= 4 kHz)
    ])
    return {"mfcc": _cepstra(power, MEL_FB), "lfcc": _cepstra(power, LIN_FB), "spec": spec}


def column_names(group):
    if group == "spec":
        rows = SPECTRAL
    else:
        rows = [f"{group}_d{d}_c{c:02d}" for d in range(3) for c in range(N_CEP)]
    return [f"{r}_{s}" if group != "spec" else f"spec_{r}_{s}" for r in rows for s in STATS]


def apply_transform(y, transform=None, key=None):
    """Audio changes for E5. Test-time: None, "trim" (cut leading/trailing silence with the
    group-S VAD), "gain-12" (12 dB quieter), "gain+6" (6 dB louder, clipped to +-1).
    Training-time "augment": trim, then pad each end with 0-1 s of background noise at -80 to
    -50 dBFS, then a random gain of -12 to +6 dB (clipped). Its randomness is seeded by `key`
    (the clip id), so the augmented set is reproducible."""
    if transform is None:
        return y
    if transform == "augment":
        import zlib

        from deepvoice.silence import trim

        rng = np.random.default_rng(zlib.crc32(str(key).encode()))
        y = trim(y)
        pads = [rng.normal(0, 10 ** (rng.uniform(-80, -50) / 20), int(rng.uniform(0, 1) * SR))
                for _ in range(2)]
        y = np.concatenate([pads[0], y, pads[1]]) * 10 ** (rng.uniform(-12, 6) / 20)
        return np.clip(y, -1, 1).astype(np.float32)
    if transform == "trim":
        from deepvoice.silence import trim

        return trim(y)
    if transform == "gain-12":
        return (y * 10 ** (-12 / 20)).astype(np.float32)
    if transform == "gain+6":
        return np.clip(y * 10 ** (6 / 20), -1, 1).astype(np.float32)
    raise ValueError(f"unknown transform {transform!r}")


def clip_features(path, transform=None):
    """Pooled feature vectors {group: 1-D float32 array} for one audio file."""
    f = frame_features(apply_transform(load_audio(path), transform, key=Path(path).stem))
    return {g: _pool(f[g]).astype(np.float32) for g in GROUPS}


def feature_tables(df, name, root="data/features", n_jobs=-1, transform=None):
    """{group: DataFrame(file + pooled columns)} for every row of `df`, cached as
    <root>/<group>_<name>.parquet. Existing caches are loaded, never recomputed. `transform`
    (see apply_transform) changes the audio first; give such a cache its own `name`."""
    root = Path(root)
    paths = {g: root / f"{g}_{name}.parquet" for g in GROUPS}
    if all(p.exists() for p in paths.values()):
        out = {g: pd.read_parquet(p) for g, p in paths.items()}
        for g, t in out.items():
            if list(t["file"]) != list(df["file"]):
                raise ValueError(f"{paths[g]} does not match the requested clips")
        return out
    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, batch_size=32)(delayed(clip_features)(p, transform) for p in df["path"])
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for g in GROUPS:
        t = pd.DataFrame(np.stack([r[g] for r in rows]), columns=column_names(g))
        t.insert(0, "file", df["file"].to_numpy())
        t.to_parquet(paths[g])
        out[g] = t
    print(f"features {name}: {len(df)} clips in {time.time() - t0:.0f}s -> {root}/<group>_{name}.parquet")
    return out


def select(table, group, n_coef=N_CEP, deltas=2, stats=("mean", "std")):
    """Columns of a cached cepstral table for one ablation setting, e.g. 20 MFCC, no deltas."""
    keep = [f"{group}_d{d}_c{c:02d}_{s}" for d in range(deltas + 1) for c in range(n_coef)
            for s in stats]
    return table[keep]


SILENCE = ["lead_sil", "trail_sil", "sil_ratio", "duration", "energy_mean", "energy_std",
           "energy_p10", "energy_p90", "energy_max"]


def load_xy(split, groups=("lfcc",), n_coef=20, deltas=2, stats=("mean", "std"),
            root="data", variant=None):
    """One modelling table: file, label, speaker, attack + feature columns.

    `split` is "train", "dev" or "eval" (ASVspoof) or "itw" (In-the-Wild, read from
    <root>/release_in_the_wild/meta.csv). Groups follow the plan: "silence" (S), "mfcc" (M),
    "lfcc" (L), "spec" (P). Cepstral groups use `n_coef`, `deltas` and `stats`; the spectral group
    uses `stats`; silence uses all 9 columns. Feature columns are every column after the first four.
    `variant` (e.g. "trim") reads the features of transformed audio, <group>_<split>_<variant>.
    """
    root = Path(root)
    if split == "itw":
        from deepvoice.data import load_itw

        p = load_itw(root / "release_in_the_wild")
    else:
        p = pd.read_parquet(root / "protocols.parquet")
    out = p[p["split"] == split][["file", "label", "speaker", "attack"]].reset_index(drop=True)
    parts = [out]
    for g in groups:
        t = pd.read_parquet(root / "features" / f"{g}_{split}{'_' + variant if variant else ''}.parquet")
        if list(t["file"]) != list(out["file"]):
            raise ValueError(f"{g}_{split} rows do not match the protocol order")
        if g in ("mfcc", "lfcc"):
            parts.append(select(t, g, n_coef, deltas, stats))
        elif g == "spec":
            parts.append(t[[f"spec_{r}_{s}" for r in SPECTRAL for s in stats]])
        elif g == "silence":
            parts.append(t[SILENCE])
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return pd.concat(parts, axis=1)
