# DeepVoice

**Do fake-voice detectors learn fakeness, or just the dataset?** CMPS 261 course project
(AUB, Fall 2026). We train detectors of increasing complexity on ASVspoof 2019 LA, tune them,
and measure how they generalize to unseen attacks and to In-the-Wild; we test for shortcuts
(silence, loudness), calibration, and whether the pretrained model is worth its cost.

> Status: Tier 1 complete (E1-E5). Final report: [`docs/final_report/final_report.pdf`](docs/final_report/final_report.pdf).
> Single notebook: [`notebooks/DeepVoice_final.ipynb`](notebooks/DeepVoice_final.ipynb). Final model and card: [`model/`](model/).

## Results in one table

EER (%) with speaker-clustered bootstrap 95% CI. Models fit on ASVspoof train, chosen on dev, scored
once on eval and In-the-Wild after the hypotheses were frozen.

| Rung | Final model | Dev | ASVspoof eval | eval, silence trimmed | In-the-Wild |
|---|---|---|---|---|---|
| 0 | depth-3 tree on 4 silence/duration numbers | 28.0 | 19.1 [18.6, 19.5] | (not run) | 37.6 [31.4, 42.4] |
| 1 | gradient boosting on LFCC-40 + deltas (tuned) | 0.13 | 14.1 [12.8, 15.5] | 24.2 | 77.6 [67.3, 85.7] |
| 2 | frozen XLS-R-300M layer 3 + logistic regression | 0.00 | **2.4** [1.5, 3.1] | 16.9 | 42.3 [33.6, 52.8] |

Hypotheses (rules frozen in `hypotheses.md`): H1a and H1b (silence shortcut) supported; H2 (unseen
attacks harder) rule met; H3 supported in-distribution, not supported on In-the-Wild; H4 false by its
rule (every rung is near 50% error at its dev threshold on In-the-Wild, but already so close to chance
at the best threshold that the pre-registered gap is not reached). Details: `results/e4_hypotheses.csv`,
`results/e5_diffs.csv`, `results/cost_table.csv`.

## Reproduce everything

```bash
./scripts/setup.sh --dl                         # uv, Python 3.12, locked packages, tests
./scripts/download_data.sh                      # ASVspoof 2019 LA + In-the-Wild (~16 GB)
uv run --extra dl python -m deepvoice.pipeline  # all feature caches (~3 h from scratch on an M5)
uv run jupyter nbconvert --to notebook --execute notebooks/DeepVoice_final.ipynb --output-dir out
```

Every table and figure in `results/` is written by a notebook cell (seed 0, a config JSON per run in
`results/configs/`). Score your own audio with the final model:
`uv run --extra dl python -m deepvoice.predict clip.wav` (read `model/MODEL_CARD.md` first).

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
src/deepvoice/   data.py (protocols, ITW, gender), metrics.py (EER, bootstrap CIs, ECE,
                 calibration), features.py, silence.py, xlsr.py, models.py, pipeline.py (builds
                 every cache), predict.py (final-model CLI)
tests/           pytest; every metric/parser gets a small test
scripts/         setup.sh (macOS/Linux), setup.ps1 (Windows), download_data.sh
notebooks/       00_audit ... 06_cost_and_final_model, and DeepVoice_final.ipynb (all of them
                 in one notebook); analysis only, real code lives in src/ (nbstripout enabled)
docs/            PLAN.pdf, progress report, final report, executed final notebook
results/         one CSV per experiment + small figures (committed); configs/ = one JSON per run
CLAUDE.md        working rules for the team and for Claude Code
model/           final model (29 KB classifier + metadata) and MODEL_CARD.md; XLS-R itself is
                 downloaded from Hugging Face
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

Frozen in commit `1d896cd` on 2026-11-06, before any model was run on ASVspoof eval or In-the-Wild.

```
sha256  9494fc18b857575a81586bd6cd7f803c3bf023f780040d894bda5db07b00b76b  hypotheses.md
```

Check it: `shasum -a 256 hypotheses.md` (macOS) or `sha256sum hypotheses.md` (Linux).
