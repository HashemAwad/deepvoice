"""Audio loading and features: one loader for every dataset (16 kHz mono), MFCC/LFCC/spectral
features pooled over time, cached to data/ and never recomputed."""

import numpy as np
import soundfile as sf

SR = 16000


def load_audio(path, sr=SR):
    """The single loader for ASVspoof and In-the-Wild: float32 mono at `sr` Hz."""
    y, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if orig_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y.astype(np.float32)
