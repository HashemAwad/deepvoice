"""Assemble notebooks/DeepVoice_final.ipynb (the single Moodle notebook) from notebooks 00-06.

    uv run python scripts/build_final_notebook.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PARTS = [
    ("00_audit", "1. Data audit",
     "Verified split counts, class balance, duration and silence by class (train and dev). Figure 1 of the report."),
    ("01_features", "2. E2: feature engineering",
     "Feature-group, MFCC-vs-LFCC, coefficient, delta and pooling ablations with fixed untuned models (dev only)."),
    ("02_baselines", "3. E3: tuning, overfitting and split sensitivity",
     "Random search on LFCC-40, selection on dev vs speaker-grouped CV, gap table and learning curves."),
    ("03_xlsr", "4. Rung 2: frozen XLS-R-300M",
     "Layer sweep on dev, small MLP, the pre-registered selection rule, and the single eval run of the chosen model."),
    ("04_generalization", "5. E4: generalization",
     "Rung 0 per attack, then all three final models refit from code and scored on dev, eval and In-the-Wild; H3 and H4 verdicts."),
    ("05_shortcuts_calibration", "6. E5: shortcuts, mitigation, calibration, failure analysis",
     "Silence trimming and gain tests (H1b), augmentation retraining, ECE and temperature scaling, failures by duration, silence, gender and speaker."),
    ("06_cost_and_final_model", "7. E1 cost table and the final model",
     "Extraction, training and scoring cost per rung; saving and checking the final model."),
]
INTRO = """# DeepVoice: Do Fake-Voice Detectors Learn Fakeness, or Just the Dataset?

**CMPS 261 Introduction to Machine Learning, AUB, Fall 2026.** Hashem Awad, Ahmed Faleh, Ali Sleiman.
Report: submitted on Moodle. Code: https://github.com/HashemAwad/deepvoice

**Question.** What do fake-voice detectors learn, and how much of their performance survives when the
data change? We train a ladder of detectors on ASVspoof 2019 LA (rung 0: silence/duration only;
rung 1: tuned classical models on LFCC; rung 2: frozen XLS-R-300M + logistic regression), test them on
unseen attacks and on In-the-Wild, and probe shortcuts, calibration and failures.

**Rules that every cell follows** (see `CLAUDE.md`): fit on train only; choose settings and thresholds
on dev; eval and In-the-Wild were used only after `hypotheses.md` was frozen and hashed (commit
`1d896cd`, sha256 in the README); primary metric EER with a speaker-clustered bootstrap 95% CI;
features computed once and cached; seed 0 and a config file for every run.

**How to run.** `./scripts/setup.sh --dl`, `./scripts/download_data.sh` (about 16 GB), then
`uv run --extra dl python -m deepvoice.pipeline` (builds all caches; about 3 h from scratch, seconds
when cached), then run this notebook top to bottom. Each section below is one of the repository's
notebooks 00-06, unchanged. The reusable code they import (audio loader, features, XLS-R embedding,
metrics, models, final-model CLI) lives in `src/deepvoice/` and is reproduced in full in the
**Appendix** at the end of this notebook."""

APPENDIX = """---
# Appendix: source code of `src/deepvoice/`

The cells above import these modules. They are reproduced here verbatim (generated from the
repository by `scripts/build_final_notebook.py`), so this notebook contains all the code used:
data loading and cleaning (`data.py`), the shared 16 kHz mono audio loader and feature extraction
(`features.py`, `silence.py`), the frozen XLS-R embeddings (`xlsr.py`), metrics with bootstrap CIs
and calibration (`metrics.py`), models, tuning and model selection (`models.py`), the cache pipeline
(`pipeline.py`) and the final-model scorer (`predict.py`)."""


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(INTRO), md("## 0. Build (or load) every cached feature set"),
             code('from pathlib import Path\n\nfrom deepvoice.pipeline import build_all\n\n'
                  'build_all("../data" if Path.cwd().name == "notebooks" else "data")')]
    for name, title, blurb in PARTS:
        src = nbf.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        cells.append(md(f"---\n# {title}\n\n{blurb} *(Source: `notebooks/{name}.ipynb`.)*"))
        for c in src.cells:
            if c.cell_type == "code":
                cells.append(code(c.source))
            else:
                cells.append(md("#" + c.source if c.source.startswith("# ") else c.source))
    cells.append(md(APPENDIX))
    for f in sorted((ROOT / "src" / "deepvoice").glob("*.py")):
        cells.append(md(f"### `src/deepvoice/{f.name}`\n\n```python\n{f.read_text().rstrip()}\n```"))
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(nb, ROOT / "notebooks" / "DeepVoice_final.ipynb")
    print(len(cells), "cells")


if __name__ == "__main__":
    main()
