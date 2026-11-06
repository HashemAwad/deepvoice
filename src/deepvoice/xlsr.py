"""Frozen XLS-R-300M embeddings (rung 2): mean-pooled hidden states every 3rd layer, fp16, cached.

The model is never trained here. Each clip goes through it once; for layers 0, 3, ..., 24 we
average the frame vectors over time (ignoring padding) and keep that 1024-dim vector.
"""

import time
from pathlib import Path

import numpy as np
import torch

from deepvoice.features import load_audio

MODEL = "facebook/wav2vec2-xls-r-300m"
LAYERS = list(range(0, 25, 3))  # 0 = CNN feature projection output, 24 = last transformer layer


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev=None):
    from transformers import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(MODEL)
    return model.eval().to(dev or device())


def _normalize(y):
    # what Wav2Vec2FeatureExtractor does: zero mean, unit variance per clip
    return (y - y.mean()) / np.sqrt(y.var() + 1e-7)


@torch.inference_mode()
def embed_batch(model, waves, dev=None):
    """Mean-pooled embeddings for a list of 16 kHz clips -> float16 array (n, len(LAYERS), 1024)."""
    dev = dev or next(model.parameters()).device
    lens = [len(w) for w in waves]
    x = np.zeros((len(waves), max(lens)), dtype=np.float32)
    mask = np.zeros_like(x, dtype=np.int64)
    for i, w in enumerate(waves):
        x[i, : len(w)] = _normalize(w)
        mask[i, : len(w)] = 1
    out = model(torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(mask).to(dev),
                output_hidden_states=True)
    frames = model._get_feat_extract_output_lengths(torch.tensor(lens)).to(dev)
    fmask = (torch.arange(out.hidden_states[0].shape[1], device=dev)[None] < frames[:, None])
    fmask = fmask[..., None].float()
    pooled = [(out.hidden_states[k] * fmask).sum(1) / fmask.sum(1) for k in LAYERS]
    return torch.stack(pooled, 1).half().cpu().numpy()


def embed_paths(model, paths, batch_seconds=60.0):
    """Embeddings for many files. Clips are sorted by length and batched up to ~batch_seconds of
    audio per batch (less padding, fewer passes); output keeps the input order."""
    waves = [load_audio(p) for p in paths]
    order = np.argsort([len(w) for w in waves])
    out = np.zeros((len(waves), len(LAYERS), 1024), dtype=np.float16)
    batch = []
    for i in order:
        batch.append(i)
        if len(waves[i]) * len(batch) / 16000 >= batch_seconds or i == order[-1]:
            out[batch] = embed_batch(model, [waves[j] for j in batch])
            batch = []
    return out


def time_it(paths, model=None, batch_seconds=60.0):
    """Seconds per clip and per audio-second for embedding `paths` (model load excluded)."""
    model = model or load_model()
    embed_paths(model, paths[:4], batch_seconds)  # warm-up: first mps call compiles kernels
    if model.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    emb = embed_paths(model, paths, batch_seconds)
    if model.device.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    audio_s = sum(len(load_audio(p)) for p in paths) / 16000
    return {"clips": len(paths), "seconds": dt, "s_per_clip": dt / len(paths),
            "audio_seconds": audio_s, "realtime_factor": audio_s / dt, "emb": emb}


def cache_path(name, root="data/features"):
    return Path(root) / f"xlsr300m_{name}.npy"


def extract_cached(df, name, root="data/features", chunk=2000, model=None):
    """Embeddings for every row of `df` -> <root>/xlsr300m_<name>.npy (n, 9 layers, 1024), fp16,
    rows in `df` order, plus <name>_files.txt listing the clip ids. Work is saved per chunk in
    <root>/xlsr300m_<name>_parts/, so an interrupted run resumes where it stopped."""
    out = cache_path(name, root)
    files = out.with_name(f"xlsr300m_{name}_files.txt")
    if out.exists():
        if files.read_text().split() != list(df["file"]):
            raise ValueError(f"{out} does not match the requested clips")
        return np.load(out, mmap_mode="r")
    parts = out.with_name(f"xlsr300m_{name}_parts")
    parts.mkdir(parents=True, exist_ok=True)
    paths = list(df["path"])
    n_chunks = (len(paths) + chunk - 1) // chunk
    t0 = time.time()
    for c in range(n_chunks):
        part = parts / f"{c:04d}.npy"
        if part.exists():
            continue
        model = model or load_model()
        np.save(part, embed_paths(model, paths[c * chunk:(c + 1) * chunk]))
        print(f"{name}: chunk {c + 1}/{n_chunks} done, {time.time() - t0:.0f}s", flush=True)
    emb = np.concatenate([np.load(parts / f"{c:04d}.npy") for c in range(n_chunks)])
    np.save(out, emb)
    files.write_text("\n".join(df["file"]) + "\n")
    return emb
