# CLAUDE.md: rules for working in this repo

DeepVoice, CMPS 261 (intro ML) team project: Hashem Awad, Ahmed Faleh, Ali Sleiman.
We train fake-voice detectors of increasing complexity on ASVspoof 2019 LA and test how they
generalize (unseen attacks, In-the-Wild), whether they use shortcuts, and whether their
confidence can be trusted. The internal plan (PDF, team only) defines Tiers 1-3, rungs 0-4,
experiments E1-E10 and hypotheses H1-H5. Finish Tier 1 before anything else.

## Hard rules (never break these; ask the team before changing any of them)

1. **No leakage.** Fit models only on train. Choose hyperparameters and thresholds on dev
   (or speaker-grouped CV on train). Do NOT run anything on ASVspoof **eval** or
   **In-the-Wild** until `hypotheses.md` is finalized, hashed (sha256), committed, and the hash
   is in `README.md`. Downloading/unzipping ITW is fine; loading it into a model is not.
2. **Same audio loader for both datasets.** Every clip is resampled to 16 kHz mono.
3. **Primary metric is EER**, not accuracy (the data is ~90% fake). Report it with a
   speaker-clustered bootstrap 95% CI (`deepvoice.metrics.speaker_bootstrap_eer`) where relevant.
   Label convention: 1 = fake, 0 = real; higher score = more fake.
4. **Compute features once and cache them to disk** under `data/` (gitignored). Never recompute
   a cached feature set; load it.
5. **Reproducible runs.** Fixed random seed and a config file per run (saved as JSON under
   `results/configs/`). One results CSV per experiment in `results/` with columns
   `rung, model, config, seed, dataset, split, eer, ci_low, ci_high, train_time, latency`
   (the CI is split into its two ends; `config` is the config file name).
6. **Tooling.** uv with Python 3.12 (`.python-version`). Run everything with `uv run`.
   Repo layout as in the plan's Section 7 (see README). Reusable code in `src/deepvoice/`,
   notebooks only for analysis.
7. **Downloads.** Never download anything large without stating its size first and checking
   free disk space (`df -h`).

## Practical notes

- Machine: MacBook, Apple M5, 32 GB RAM, no NVIDIA GPU. Neural work uses torch device `"mps"`
  with `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically when `deepvoice` is imported first).
- Data lives in `data/LA/` and `data/release_in_the_wild/` (see `scripts/download_data.sh`).
  Iterate over protocol files, never glob the flac folders (they hold extra ASV files).
- Git: work on a short branch, open a PR, a teammate reviews; do not push straight to `main`.
- **Git author per session:** all three of us share this laptop. At the start of each session,
  ask who is working and set it for this repo:
  `git config user.name "<Name>"` and `git config user.email "<email>"`.
- Teaching mode: the team is new to ML. Before each step explain in 2-4 plain sentences what
  and why, define each new term once, report what results mean, stop at checkpoints, and never
  silently skip or change anything in the plan.
