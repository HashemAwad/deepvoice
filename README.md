# DeepVoice

**Do fake-voice detectors learn fakeness, or just the dataset?** CMPS 261 course project
(AUB, Fall 2026). We train detectors of increasing complexity on ASVspoof 2019 LA, tune them,
and measure how they generalize to unseen attacks and to In-the-Wild; we test for shortcuts
(silence, loudness), calibration, and whether the pretrained model is worth its cost.

> Status: early setup. Hypotheses file is a **draft** until frozen (see below).

## Setup

```bash
uv sync                  # core deps (CPU: features, sklearn, metrics)
uv sync --extra dl       # on the GPU machine: torch, transformers, peft, optuna
uv run pytest            # should pass before you push
```

Python 3.11 or 3.12. Run everything through `uv run`.

## Layout

```
src/deepvoice/   data.py (protocols), metrics.py (EER, bootstrap CI, ECE); add features.py,
                 silence.py, xlsr.py, models.py as we go
tests/           pytest; every metric/parser gets a small test
notebooks/       analysis only; real code lives in src/ (run `nbstripout --install` once)
results/         one CSV per experiment + small figures (committed)
model/           final model card; weights go to a release / HF Hub, not git
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

- ASVspoof 2019 LA: Edinburgh DataShare, doi 10.7488/ds/2555 (LA.zip, ~7.1 GB). Put it under
  `data/LA/`. Verify protocol/folder names in `src/deepvoice/data.py` against the real files.
- In-the-Wild: Hugging Face `mueller91/In-The-Wild`.

## Hypotheses hash

Not frozen yet. When frozen, record `sha256sum hypotheses.md` here.
