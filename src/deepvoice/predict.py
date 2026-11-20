"""Score audio files with the final DeepVoice model (rung 2: frozen XLS-R-300M layer 3 + logistic
regression).

    uv run --extra dl python -m deepvoice.predict clip1.wav clip2.flac

Prints P(fake) and the decision at the ASVspoof-dev EER threshold. Read model/MODEL_CARD.md
first: on real-world audio (In-the-Wild) this model is close to chance.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
MODEL_FILE = "rung2_xlsr300m_l3_logreg.joblib"


def load(model_dir=MODEL_DIR):
    """(classifier, metadata) of the saved final model."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / MODEL_FILE.replace(".joblib", ".json")).read_text())
    return joblib.load(model_dir / MODEL_FILE), meta


def score(paths, clf=None, meta=None, xlsr_model=None):
    """P(fake) for each audio file (any sample rate; resampled to 16 kHz mono)."""
    from deepvoice import xlsr

    if clf is None:
        clf, meta = load()
    xlsr_model = xlsr_model or xlsr.load_model()
    emb = xlsr.embed_paths(xlsr_model, [str(p) for p in paths])
    x = emb[:, xlsr.LAYERS.index(meta["layer"])].astype(np.float32)
    return clf.predict_proba(x)[:, 1]


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    clf, meta = load()
    for p, s in zip(paths, score(paths, clf, meta)):
        label = "fake" if s >= meta["threshold"] else "real"
        print(f"{p}\tP(fake)={s:.3f}\t{label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
