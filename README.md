# DeepVoice

**Do fake-voice detectors learn fakeness, or just the dataset?** CMPS 261 course project
(AUB, Fall 2026). We train detectors of increasing complexity on ASVspoof 2019 LA, tune them,
and measure how they generalize to unseen attacks and to In-the-Wild; we test for shortcuts
(silence, loudness), calibration, and whether the pretrained model is worth its cost.

> Status: early setup. Hypotheses file is a **draft** until frozen (see below).

## Setup

Prerequisite: **git**. Everything else (uv, Python 3.12, all packages) is installed by the script.

| OS | Command (from the repo root) |
|---|---|
| macOS / Linux | `./scripts/setup.sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1` |

Add `--dl` (macOS/Linux) or `-Dl` (Windows) for the deep-learning stack (torch, transformers,
peft, optuna). The script installs `uv` if missing, installs Python 3.12, syncs the locked
dependencies, enables notebook output stripping, and runs the tests.

- **GPU work belongs on the Linux compute machine**: there `uv sync --extra dl` pulls
  CUDA-enabled torch. On Mac, torch uses Apple's MPS backend. On Windows, PyPI torch is CPU-only.
- Manual equivalent: `uv sync` (add `--extra dl` if needed), `uv run nbstripout --install`,
  `uv run pytest`.
- Run everything through `uv run`, e.g. `uv run python -m ...`. The Python version is pinned
  in `.python-version`.

## Layout

```
src/deepvoice/   data.py (protocols), metrics.py (EER, bootstrap CI, ECE); add features.py,
                 silence.py, xlsr.py, models.py as we go
tests/           pytest; every metric/parser gets a small test
scripts/         setup.sh (macOS/Linux) and setup.ps1 (Windows)
notebooks/       analysis only; real code lives in src/ (setup script enables nbstripout)
results/         one CSV per experiment + small figures (committed)
model/           final model card; large weights go to a release / HF Hub, not git
                 (a small final model such as an sklearn .joblib may be committed)
hypotheses.md    frozen + hashed before any eval run
data/ cache/     GITIGNORED: raw audio, features, checkpoints
```

## Rules

1. **Never commit audio, features or checkpoints.** Share them via the team drive / HF Hub.
2. **Eval and In-the-Wild are touched only after `hypotheses.md` is frozen**, and reported once
   per final model. Tune on dev (or speaker-grouped CV on train) only.
3. Label convention everywhere: **1 = fake, 0 = real; higher score = more fake.**
4. Every run has a seed and a config; results go to `results/<experiment>.csv` with columns
   `rung, model, config, seed, dataset, split, eer, ci_low, ci_high, train_time, latency`.
5. Work on short branches, open a PR, one teammate reviews. Do not push straight to `main`
   after the first commit.
6. Notebooks hold analysis, not logic. Reusable code goes in `src/deepvoice/`.

## Data (download on the compute machine, not in git)

Run on the machine that trains (needs `wget`, `unzip`, ~35 GB free at peak, ~16 GB after):

```bash
./scripts/download_data.sh    # resumable: rerun if the connection drops
```

| Dataset | Source | Download | Ends up at |
|---|---|---|---|
| ASVspoof 2019 LA | Edinburgh DataShare, doi 10.7488/ds/2555 (ODC-By; see `LICENSE.txt` inside each split folder) | `LA.zip`, 7,640,952,520 bytes | `data/LA/` (122,299 flac) |
| In-the-Wild | Hugging Face `mueller91/In-The-Wild` (public, no login; HF tags CC-BY-SA-4.0) | `release_in_the_wild.zip`, 8,161,489,023 bytes | `data/release_in_the_wild/` (31,779 wav + `meta.csv`) |

Facts checked against the real files (2026-11-05):
- LA protocols: train 2,580 real + 22,800 fake, **20 speakers**, attacks A01-A06; dev 2,548 +
  22,296, **20 speakers**, A01-A06; eval 7,355 + 63,882, **67 speakers**, A07-A19 (4,914 clips
  per attack). Speakers do not overlap across splits.
- A16 and A19 are listed in `README.LA.txt` with the same system types as A04 (TTS, waveform
  concatenation) and A06 (VC, spectral filtering), consistent with the paper saying they
  repeat training attacks. Treat as **repeated**; the unseen eval attacks are A07-A15, A17, A18.
- ITW `meta.csv` has columns `file, speaker, label` (labels `bona-fide` / `spoof`; 19,963 +
  11,816). It has **54 distinct speaker names**, not the 58 quoted in the paper.
- Keep raw audio **on a fast local disk of the compute machine**, not on a synced or network
  drive (reading 150k small files from Drive/NFS is very slow). If several of us share one
  machine, keep one copy and symlink `data` to it. Share *derived* files (features,
  embeddings, checkpoints) through a private Hugging Face repo or the team drive.

## Hypotheses hash

Not frozen yet. When frozen, record `sha256sum hypotheses.md` here.
