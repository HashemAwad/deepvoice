# Model card: DeepVoice rung 2 (frozen XLS-R-300M + logistic regression)

CMPS 261 course project (AUB, Fall 2026): Hashem Awad, Ahmed Faleh, Ali Sleiman.
This is a **research artifact for a study of what fake-voice detectors learn**. It is **not** a
deployable deepfake detector; read "Limitations" before using it on any audio.

## What it is

| | |
|---|---|
| Input | one audio file, any sample rate; resampled to 16 kHz mono |
| Features | `facebook/wav2vec2-xls-r-300m` (frozen, 315M parameters, not trained by us); hidden states of transformer layer 3, averaged over time (1,024 numbers) |
| Classifier | standardization + logistic regression (C = 1, class-balanced), 1,025 trained parameters |
| Output | P(fake); called "fake" if P(fake) >= 0.528 (the equal-error threshold on ASVspoof 2019 LA dev) |
| Files | `rung2_xlsr300m_l3_logreg.joblib` (29 KB) and `.json` (metadata, sha256 `701d346a...cbbc`) |
| Code | `src/deepvoice/predict.py`; reproduced by `notebooks/06_cost_and_final_model.ipynb` |

```bash
uv sync --extra dl
uv run python -m deepvoice.predict my_clip.wav
```

The first run downloads XLS-R-300M (1.27 GB) from Hugging Face.

## Training data and selection

- Trained on ASVspoof 2019 LA **train** only: 2,580 real + 22,800 fake clips, 20 speakers,
  attacks A01-A06 (studio-quality English, VCTK speakers).
- Layer and classifier were chosen on ASVspoof **dev** (lowest dev EER; ties inside the 95% CI broken
  by speaker-grouped cross-validation on train). Eval and In-the-Wild were used once, after the
  hypotheses were frozen (`hypotheses.md`, sha256 in the README).

## Results (EER, speaker-clustered bootstrap 95% CI)

| Test set | EER | Notes |
|---|---|---|
| ASVspoof 2019 LA dev | 0.00% | same attacks as training: saturated |
| ASVspoof 2019 LA eval | **2.37%** [1.53, 3.06] | 13 attacks, 11 never seen in training |
| ASVspoof eval, silence trimmed | 16.9% | +14.6 pp: the model relies on leading/trailing silence |
| In-the-Wild (real-world audio) | **42.3%** [33.6, 52.8] | close to chance (50%) |
| In-the-Wild at the dev threshold | 47.4% error | the threshold does not transfer |

Cost: about 45 ms per clip for XLS-R on an Apple M5 GPU, under 0.01 ms for the classifier.

## Limitations (please read)

- **It does not generalize to real-world audio.** On In-the-Wild it is near chance, and for 25 of 54
  speakers it is worse than chance. It flags old or noisy real recordings as fake (EER above 90% for
  JFK, Harry Truman, George W. Bush and Frank Sinatra) while doing well on modern, clean voices.
  It reacts to recording conditions as much as to fakeness.
- **It uses a dataset shortcut.** Removing the silence before and after speech raises its ASVspoof
  eval EER from 2.4% to 16.9%. Real and fake ASVspoof clips differ in leading silence, and the model
  learned that.
- **Its confidence is not calibrated outside ASVspoof dev.** Expected calibration error is 0% on
  dev, 18% on eval and 58% on In-the-Wild; temperature scaling fitted on dev does not fix it.
  Do not read P(fake) as a probability.
- **The threshold drifts.** At the dev threshold it lets about 20% of eval fakes through.
- English only, one training corpus, 20 training speakers; trained on balanced class weights
  although the data are about 90% fake.

## Intended use

Coursework and research on dataset shortcuts and generalization in spoofing detection. Not for
judging whether a real recording is genuine, for moderation, forensics, or any decision about a
person.

## Data licences

ASVspoof 2019 LA (Edinburgh DataShare, ODC-By); In-the-Wild (Hugging Face `mueller91/In-The-Wild`,
CC-BY-SA-4.0 per its card). No audio is redistributed with this model.
