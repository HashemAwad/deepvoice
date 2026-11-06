# Model card (draft)

To be filled in once the final model is chosen (Nov 13 freeze).

- **Model:** TBD (rung, features, hyperparameters)
- **Intended use:** coursework research on fake-voice detection; not a production detector.
- **Training data:** ASVspoof 2019 LA train split (16 kHz mono).
- **Evaluation:** EER with speaker-clustered bootstrap 95% CI on ASVspoof dev/eval and In-the-Wild.
- **Limitations:** trained on one corpus; In-the-Wild results reflect a bundled data-source shift.
- **Download:** TBD
