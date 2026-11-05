# Hypotheses (DRAFT: not yet frozen)

> **Rule:** this file is finalized, hashed (`sha256sum hypotheses.md`), and committed
> **before any ASVspoof eval or In-the-Wild evaluation is run.** After that commit, edits are
> only allowed as clearly dated addenda; never rewrite a claim or threshold.
>
> When frozen: change the status above, record the hash in `README.md`, and commit.

Primary metric: EER with a speaker-clustered bootstrap 95% CI. Secondary: ECE (15 bins),
error at the dev-fixed threshold vs the best threshold. 5 pp is decidable on ASVspoof eval;
only >= 10 pp is decidable on ITW (smaller, speaker-clustered).

| # | Claim | Decided true / false by |
|---|---|---|
| H1 | The dataset has a silence shortcut. | Rung 0 eval EER < 25% (false if >= 35%). Trimming silence raises rung 1 eval EER by >= 5 pp with CI excluding 0. |
| H2 | Unseen attacks are harder. | Worst unseen attack EER >= 2x the mean seen/repeated-attack EER for rung 1, after verifying which eval attacks repeat training attacks (A16/A19). |
| H3 | The pretrained model wins in-distribution, and its edge shrinks on new data. | Rung 2 beats best rung 1 by >= 5 pp on ASVspoof eval. On ITW the margin is reported; only >= 10 pp is decidable. |
| H4 | Dev-chosen thresholds do not survive the data change. | On ITW, error at the dev threshold is >= 10 pp above best-threshold error for at least 2 of 3 rungs. |
| H5 | (Tier 2; keep only if E6 will run) Fine-tuning helps in-distribution more than out-of-distribution. | Fine-tuned minus frozen EER gain on ASVspoof eval exceeds the gain on ITW by >= 5 pp, across seeds. Report the opposite if so. |

Results are reported whether or not they match.
