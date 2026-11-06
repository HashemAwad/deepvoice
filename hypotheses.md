# Hypotheses (FROZEN before any ASVspoof eval or In-the-Wild run)

> **Rule:** this file was finalized, hashed (`shasum -a 256 hypotheses.md`) and committed
> **before any model was run on ASVspoof eval or In-the-Wild.** The hash is in `README.md`.
> After that commit, changes are only allowed as clearly dated addenda at the end; never rewrite a
> claim or threshold. Results are reported whether or not they match.

State at freeze: models have been fit on ASVspoof 2019 LA train and scored on train and dev only
(rung 0, silence/duration). For eval, only the protocol file (labels) has been read; no eval or
In-the-Wild audio has been loaded into any feature extractor or model.

## Definitions

- **EER**: `deepvoice.metrics.eer` (ROC-interpolated equal error rate; label 1 = fake). CIs are
  speaker-clustered bootstrap 95% intervals (`speaker_bootstrap_eer`, 1000 resamples). A
  difference between two EERs on the same clips uses a paired bootstrap (the same speaker
  resamples for both) and "CI excluding 0" refers to that interval.
- **pp** = percentage points (30% to 35% is +5 pp).
- **Rung model**: for each rung, the model with the lowest **dev** EER, chosen before eval.
  Rung 0 = silence/duration only; rung 1 = classical features (MFCC/LFCC/spectral) with
  LR/RF/HGB; rung 2 = frozen XLS-R-300M with a light classifier.
- **Attacks on eval**: per the ASVspoof 2019 paper (Wang et al. 2020, Sec. 3.1), A16 and A19 reuse
  the algorithms of training attacks A04 and A06 (**repeated**); A07-A15, A17, A18 are the 11
  **unseen** attacks. A per-attack EER uses that attack's fakes against all eval real clips.
- **Error at a threshold** = the average of the false-accept and false-reject rates at that
  threshold (half total error rate). **Best-threshold error** on a set = its EER.
- **Dev threshold** = the EER threshold of that model on ASVspoof dev.

5 pp is decidable on ASVspoof eval (large). In-the-Wild (ITW) is smaller and speaker-clustered,
so only differences >= 10 pp are decidable there.

## Hypotheses

| # | Claim | Decided true / false by |
|---|---|---|
| H1 | The dataset has a silence shortcut. | **H1a:** rung 0 eval EER < 25% is true; >= 35% is false; in between is inconclusive. **H1b:** trimming leading/trailing silence from eval raises the rung 1 eval EER by >= 5 pp with the CI excluding 0 (true), otherwise false. H1a and H1b are reported separately. |
| H2 | Unseen attacks are harder. | For the rung 1 model: the worst unseen-attack EER is >= 2x the mean EER of the repeated attacks (A16, A19). True if so, otherwise false. |
| H3 | The pretrained model wins in-distribution, and its edge shrinks on new data. | Rung 2 beats the rung 1 model by >= 5 pp on ASVspoof eval (true; otherwise false). On ITW the rung 2 minus rung 1 margin is reported; a shrinkage is only claimed if it is >= 10 pp. |
| H4 | Dev-chosen thresholds do not survive the data change. | On ITW, error at the dev threshold is >= 10 pp above the best-threshold error for at least 2 of the 3 rungs (0, 1, 2). True if so, otherwise false. |

H5 (fine-tuning, Tier 2) from the plan is **not** pre-registered: it applies only if experiment E6
runs, which is not planned on the current hardware. If E6 runs, its results are reported as
exploratory, not as a test of a pre-registered hypothesis.
